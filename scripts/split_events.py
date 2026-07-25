from pathlib import Path
from collections import defaultdict
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KUBERNETES_DATA_DIR = DATA_DIR / "kubernetes"

INPUT_FILE = KUBERNETES_DATA_DIR / "events/warning-events.yaml"
OUTPUT_DIR = KUBERNETES_DATA_DIR / "events/grouped_events"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading YAML...")

with open(INPUT_FILE, "r", encoding="utf-16") as f:
    data = yaml.safe_load(f)

groups = defaultdict(list)

for event in data.get("items", []):
    obj = event.get("involvedObject", {})

    key = (
        obj.get("namespace", "unknown"),
        obj.get("kind", "unknown"),
        obj.get("name", "unknown"),
    )

    groups[key].append(event)

print(f"Found {len(groups):,} object groups")

for namespace, kind, name in groups:
    events = groups[(namespace, kind, name)]

    safe_name = (
        f"{namespace}_{kind}_{name}"
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )

    output = {
        "namespace": namespace,
        "kind": kind,
        "name": name,
        "event_count": len(events),
        "events": events,
    }

    output_file = OUTPUT_DIR / f"{safe_name}.yaml"

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(
            output,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

print(f"Wrote {len(groups):,} files")