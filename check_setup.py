import os
import re
import sys

from dotenv import dotenv_values

CHECK = "✅"
FAIL = "❌"


def print_result(label: str, ok: bool, detail: str | None = None) -> bool:
    icon = CHECK if ok else FAIL
    if detail:
        print(f"{icon} {label} ({detail})")
    else:
        print(f"{icon} {label}")
    return ok


def normalize_requirement_name(requirement_line: str) -> str:
    base = re.split(r"[<>=!~]", requirement_line, maxsplit=1)[0]
    return base.strip()


def module_name_for_package(package_name: str) -> str:
    mapping = {
        "open-clip-torch": "open_clip",
        "opencv-python": "cv2",
        "beautifulsoup4": "bs4",
        "python-dotenv": "dotenv",
        "pillow": "PIL",
        "tavily-python": "tavily",
    }
    return mapping.get(package_name, package_name.replace("-", "_"))


def read_requirements(requirements_path: str) -> list[str]:
    packages: list[str] = []
    with open(requirements_path, "r", encoding="utf-8") as req_file:
        for raw_line in req_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            packages.append(normalize_requirement_name(line))
    return packages


def check_env_keys(root_dir: str) -> bool:
    print("\nEnvironment keys")
    env_path = f"{root_dir}/.env"
    env_values = dotenv_values(env_path)

    ok_serper = print_result(
        "SERPER_API_KEY set",
        bool(str(env_values.get("SERPER_API_KEY", "")).strip()),
    )
    ok_tavily = print_result(
        "TAVILY_API_KEY set",
        bool(str(env_values.get("TAVILY_API_KEY", "")).strip()),
    )
    return ok_serper and ok_tavily


def check_imports(requirements_path: str) -> bool:
    print("\nPackage imports")
    all_ok = True
    for package in read_requirements(requirements_path):
        module_name = module_name_for_package(package)
        try:
            __import__(module_name)
            print_result(f"{package} importable", True)
        except Exception as exc:
            all_ok = False
            print_result(f"{package} importable", False, str(exc))
    return all_ok


def main() -> int:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(root_dir, "requirements.txt")

    print("Veritas v2 setup check")
    env_ok = check_env_keys(root_dir)
    imports_ok = check_imports(requirements_path)

    if env_ok and imports_ok:
        print("\n✅ All critical checks passed")
        return 0

    print("\n❌ One or more critical checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
