import contextlib
import fnmatch
import io
from pathlib import Path
import sys
import tempfile
import unittest

import deploy_to_hf


# This is intentionally not a real credential and never reaches a network API.
TEST_TOKEN = "unit-test-client-input"


class FakeHfApi:
    def __init__(self, token, username="demo-user", error=None):
        self.token = token
        self.username = username
        self.error = error
        self.uploads = []
        self.selected_files = []

    def whoami(self):
        if self.error is not None:
            raise self.error
        return {"name": self.username, "type": "user"}

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        root = Path(kwargs["folder_path"])
        patterns = kwargs["allow_patterns"]
        self.selected_files = sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file()
            and any(fnmatch.fnmatchcase(str(path.relative_to(root)), pattern) for pattern in patterns)
        )


class DeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-deploy-", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename in deploy_to_hf.PUBLIC_UPLOAD_FILES:
            (self.root / filename).write_text("public test fixture\n")
        (self.root / ".env").write_text("PRIVATE_FIXTURE_VALUE\n")
        (self.root / "model_fp32.pth").write_bytes(b"generated test fixture")
        (self.root / "notes.docx").write_bytes(b"private test fixture")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("private test fixture\n")
        self.calls = []
        self.api = None

    def environ(self):
        return {"HF_TOKEN": TEST_TOKEN, "HF_SPACE_ID": "demo-user/existing-space"}

    def factory(self, **kwargs):
        self.calls.append(kwargs)
        self.api = FakeHfApi(**kwargs)
        return self.api

    def test_missing_token_stops_before_api_creation(self):
        with self.assertRaisesRegex(deploy_to_hf.DeploymentError, "HF_TOKEN"):
            deploy_to_hf.deploy({"HF_SPACE_ID": "demo-user/existing-space"}, self.factory, self.root)
        self.assertEqual(self.calls, [])

    def test_placeholders_stop_before_api_creation(self):
        # TEST ONLY: assemble a synthetic repeated-character placeholder, never a credential.
        repeated_placeholder = "hf_" + ("T" * 32)
        for placeholder in ("your_hf_token_here", "hf_your_token_here", "__REMOVED_HUGGINGFACE_TOKEN__", repeated_placeholder):
            with self.subTest(placeholder_type=placeholder.split("_")[0]):
                env = self.environ()
                env["HF_TOKEN"] = placeholder
                with self.assertRaisesRegex(deploy_to_hf.DeploymentError, "placeholder"):
                    deploy_to_hf.deploy(env, self.factory, self.root)
        self.assertEqual(self.calls, [])

    def test_missing_invalid_or_placeholder_space_stops_before_api(self):
        for space_id in ("", "invalid", "../space", "demo-user/space?token=value", "your-username/your-existing-space"):
            env = self.environ()
            env["HF_SPACE_ID"] = space_id
            with self.assertRaises(deploy_to_hf.DeploymentError):
                deploy_to_hf.deploy(env, self.factory, self.root)
        self.assertEqual(self.calls, [])

    def test_personal_space_upload_selects_only_the_public_allowlist(self):
        result = deploy_to_hf.deploy(self.environ(), self.factory, self.root)
        self.assertEqual(result, "demo-user/existing-space")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.api.selected_files, ["README.md", "app.py", "requirements.txt"])
        self.assertEqual(len(self.api.uploads), 1)
        upload = self.api.uploads[0]
        self.assertEqual(upload["repo_id"], "demo-user/existing-space")
        self.assertEqual(upload["repo_type"], "space")
        self.assertEqual(upload["folder_path"], str(self.root.resolve()))

    def test_other_user_or_organization_namespace_never_uploads(self):
        for namespace in ("another-user", "an-organization"):
            env = self.environ()
            env["HF_SPACE_ID"] = f"{namespace}/existing-space"
            with self.assertRaisesRegex(deploy_to_hf.DeploymentError, "個人命名空間"):
                deploy_to_hf.deploy(env, self.factory, self.root)
            self.assertEqual(self.api.uploads, [])

    def test_missing_public_file_stops_before_api_creation(self):
        (self.root / "README.md").unlink()
        with self.assertRaisesRegex(deploy_to_hf.DeploymentError, "必要檔案"):
            deploy_to_hf.deploy(self.environ(), self.factory, self.root)
        self.assertEqual(self.calls, [])

    def test_symlinked_public_file_stops_before_api_creation(self):
        (self.root / "README.md").unlink()
        (self.root / "README.md").symlink_to(self.root / ".env")
        with self.assertRaisesRegex(deploy_to_hf.DeploymentError, "符號連結"):
            deploy_to_hf.deploy(self.environ(), self.factory, self.root)
        self.assertEqual(self.calls, [])

    def test_sdk_exception_details_and_token_are_not_printed(self):
        def failing_factory(**kwargs):
            return FakeHfApi(**kwargs, error=RuntimeError(f"private SDK detail: {TEST_TOKEN}"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = deploy_to_hf.main(self.environ(), failing_factory, self.root)
        self.assertEqual(result, 1)
        self.assertIn("部署失敗", output.getvalue())
        self.assertNotIn(TEST_TOKEN, output.getvalue())
        self.assertNotIn("private SDK detail", output.getvalue())

    def test_success_output_contains_url_and_no_token(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = deploy_to_hf.main(self.environ(), self.factory, self.root)
        self.assertEqual(result, 0)
        self.assertIn("https://huggingface.co/spaces/demo-user/existing-space", output.getvalue())
        self.assertNotIn(TEST_TOKEN, output.getvalue())

    def test_import_and_fake_api_tests_do_not_load_the_real_sdk(self):
        self.assertNotIn("huggingface_hub", sys.modules)


if __name__ == "__main__":
    unittest.main()
