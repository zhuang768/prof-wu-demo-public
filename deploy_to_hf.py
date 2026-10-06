"""Upload the public demo files to an existing, personally owned HF Space."""

import os
from pathlib import Path
import re
import sys
from typing import Callable, Mapping, Optional

PUBLIC_UPLOAD_FILES = ("app.py", "requirements.txt", "README.md")
PROJECT_ROOT = Path(__file__).resolve().parent


class DeploymentError(ValueError):
    """A configuration error with a message that contains no credentials."""


def read_configuration(environ: Mapping[str, str]):
    token = environ.get("HF_TOKEN", "").strip()
    if not token:
        raise DeploymentError("請設定 HF_TOKEN；不要把存取 token 寫入程式或版本管理。")
    lowered = token.lower()
    placeholder_markers = ("your_hf_token", "your_token", "placeholder", "replace_me", "__removed_")
    suffix = token[3:] if token.startswith("hf_") else token
    if any(marker in lowered for marker in placeholder_markers) or (
        token.startswith("hf_") and len(set(suffix)) <= 2
    ):
        raise DeploymentError("HF_TOKEN 仍是範例 placeholder；請在執行環境設定自己的 token。")

    space_id = environ.get("HF_SPACE_ID", "").strip()
    if not space_id:
        raise DeploymentError("請設定 HF_SPACE_ID，格式為你自己的使用者名稱/既有Space名稱。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", space_id):
        raise DeploymentError("HF_SPACE_ID 格式必須為使用者名稱/Space名稱。")
    if space_id.lower().startswith("your-username/"):
        raise DeploymentError("HF_SPACE_ID 仍是範例；請指定你自己的既有 Space。")
    return token, space_id


def validate_public_files(project_root: Path):
    for filename in PUBLIC_UPLOAD_FILES:
        source = project_root / filename
        if source.is_symlink() or not source.is_file():
            raise DeploymentError("公開上傳清單中的必要檔案缺失或為符號連結；已停止部署。")


def deploy(
    environ: Optional[Mapping[str, str]] = None,
    api_factory: Optional[Callable] = None,
    project_root: Optional[Path] = None,
):
    token, space_id = read_configuration(os.environ if environ is None else environ)
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    validate_public_files(root)

    if api_factory is None:
        from huggingface_hub import HfApi
        api_factory = HfApi
    api = api_factory(token=token)
    identity = api.whoami()
    owner = space_id.split("/", 1)[0]
    if not isinstance(identity, dict) or identity.get("name") != owner:
        raise DeploymentError("HF_SPACE_ID 必須位於目前 token 所屬使用者的個人命名空間；已停止部署。")

    # A fixed allowlist prevents local credentials and private files from uploading.
    api.upload_folder(
        folder_path=str(root),
        repo_id=space_id,
        repo_type="space",
        allow_patterns=list(PUBLIC_UPLOAD_FILES),
    )
    return space_id


def main(environ=None, api_factory=None, project_root=None):
    try:
        space_id = deploy(environ, api_factory, project_root)
    except DeploymentError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        # SDK exception text can contain request details; never echo it or the token.
        print("部署失敗：請確認 token 權限、既有 Space 與網路設定。未輸出憑證或 SDK 回應內容。", file=sys.stderr)
        return 1
    print(f"部署完成：https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
