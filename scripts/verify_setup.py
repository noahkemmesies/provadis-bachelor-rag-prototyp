#!/usr/bin/env python
"""Verify project setup and all imports work correctly"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check if all required packages can be imported"""
    print("🔍 Checking imports...")
    print("-" * 60)

    packages = [
        ("yaml", "PyYAML"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("rank_bm25", "rank_bm25"),
        ("sentence_transformers", "sentence-transformers"),
        ("chromadb", "chromadb"),
        ("sklearn", "scikit-learn"),
        ("langchain", "langchain"),
        ("tqdm", "tqdm"),
    ]

    failed = []
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name:30} OK")
        except ImportError as e:
            print(f"❌ {package_name:30} MISSING")
            failed.append(package_name)

    return len(failed) == 0, failed


def check_local_imports():
    """Check if local modules can be imported"""
    print("\n🔍 Checking local imports...")
    print("-" * 60)

    local_modules = [
        ("src.config", "Configuration"),
        ("src.utils.logger", "Logger"),
        ("src.utils.helpers", "Helpers"),
        ("src.data.loader", "Data Loader"),
        ("src.data.preprocessor", "Preprocessor"),
        ("src.retrieval.base", "Base Retriever"),
        ("src.retrieval.bm25_retriever", "BM25 Retriever"),
        ("src.retrieval.embedding_retriever", "Embedding Retriever"),
        ("src.rag.prompting", "Prompting"),
        ("src.evaluation.metrics", "Metrics"),
    ]

    failed = []
    for module_name, display_name in local_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name:30} OK")
        except Exception as e:
            print(f"❌ {display_name:30} FAILED: {e}")
            failed.append(display_name)

    return len(failed) == 0, failed


def check_directories():
    """Check if required directories exist"""
    print("\n🔍 Checking directories...")
    print("-" * 60)

    project_root = Path(__file__).parent.parent
    directories = [
        ("src", "Source Code"),
        ("data", "Data Directory"),
        ("data/kubernetes", "Kubernetes Data"),
        ("data/queries", "Queries"),
        ("data/processed", "Processed Data"),
        ("tests", "Tests"),
        ("scripts", "Scripts"),
        ("notebooks", "Notebooks"),
        ("results", "Results"),
        ("logs", "Logs"),
    ]

    failed = []
    for dir_name, display_name in directories:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {display_name:30} exists")
        else:
            print(f"❌ {display_name:30} MISSING")
            failed.append(display_name)

    return len(failed) == 0, failed


def main():
    """Run all checks"""
    print("=" * 60)
    print("RAG Kubernetes Retrieval - Setup Verification")
    print("=" * 60)

    # Check directories
    dirs_ok, dirs_failed = check_directories()

    # Check external imports
    imports_ok, imports_failed = check_imports()

    # Check local imports
    local_ok, local_failed = check_local_imports()

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    all_ok = dirs_ok and imports_ok and local_ok

    if dirs_ok:
        print("✅ All directories present")
    else:
        print(f"❌ Missing directories: {dirs_failed}")

    if imports_ok:
        print("✅ All external packages installed")
    else:
        print(f"❌ Missing packages: {imports_failed}")
        print("\n💡 Install with: pip install -r requirements.txt")

    if local_ok:
        print("✅ All local modules importable")
    else:
        print(f"❌ Failed modules: {local_failed}")

    print("\n" + "=" * 60)

    if all_ok:
        print("🎉 Setup verification PASSED!")
        print("\n✅ Ready to proceed with Phase 2: Data Preparation")
        print("\nNext steps:")
        print("1. Add Kubernetes YAML files to data/kubernetes/")
        print("2. Create queries.json in data/queries/")
        print("3. Run: pytest tests/ -v")
        print("4. Start Phase 2!")
        return 0
    else:
        print("❌ Setup verification FAILED!")
        print("\n💡 Please fix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
