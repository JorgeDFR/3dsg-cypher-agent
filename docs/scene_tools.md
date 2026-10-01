# Scene Utilities

Utilities in this repository help prepare 3DSG inputs before dataset
generation. They do not modify the source scene unless an output path explicitly
targets it.

## Label room regions

Interactively add or update room classifications in a Spark DSG scene:

```bash
docker/stack.sh main python scripts/graph/edit_3dsg_room_labels.py \
  data/scenes/uhumans2_office.json \
  --output data/scenes/uhumans2_office.updated.json \
  --figure-output data/scenes/uhumans2_office_rooms.png
```

The editor presents a top-down view with:

- red room bounding boxes.
- room symbols.
- blue object triangles.

It visits rooms in symbol order, prints contained-object counts by class, and
accepts either a listed single-word label or its number.

### Output behavior

| Option | Behavior |
| --- | --- |
| No `--output` | Preserve the source and write `*.room_labels.json` beside it |
| `--output PATH` | Write the updated scene to `PATH` |
| `--figure-output PATH` | Save the room visualization |
| `--no-show` | Run without opening the graphical window |
| `--list-labels` | Print the controlled room vocabulary |

After changing the scene used by `configs/data.yaml`, reload it into Neo4j and
start a new dataset version. See
[Prepare a dataset version](dataset_generation.md#prepare-a-dataset-version).
