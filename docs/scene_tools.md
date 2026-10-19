# Scene Utilities

Prepare source scenes before [dataset generation](dataset_generation.md).
By default, these tools write new files beside the source.

## Label room regions

Add or update room classifications in a Spark DSG scene:

```bash
docker/stack.sh main python scripts/graph/edit_3dsg_room_labels.py \
  data/scenes/uhumans2_office.json \
  --output data/scenes/uhumans2_office.updated.json \
  --figure-output data/scenes/uhumans2_office_rooms.png
```

The editor shows room boxes, symbols, and object markers. It visits rooms in
symbol order and lists contained objects by class.

Enter a listed label or its number. Input ignores case and extra whitespace.
Press Enter to keep a current canonical label. Only listed labels are accepted.

| Option | Behavior |
| --- | --- |
| No `--output` | Write `*.room_labels.json` beside the source |
| `--output PATH` | Write the updated scene to `PATH` |
| `--figure-output PATH` | Save the room visualization |
| `--no-show` | Skip the graphical window |
| `--list-labels` | Print the room vocabulary |
| `--force` | Allow replacing an existing output, including the source |

## Repair object-to-place connections

Connect objects without a connection to their nearest mesh place:

```bash
docker/stack.sh main python scripts/graph/fix_3dsg_object_place_edges.py \
  data/scenes/uhumans2_office.json \
  --target-layer mesh-place \
  --output data/scenes/uhumans2_office.connected.json
```

Use `--target-layer place` for the place layer. Selection uses the scene's
layer mapping.

The tool keeps existing connections and adds missing edges from place to
object using distance between node positions. This is a geometric approximation.
Nearby nodes can lie across a wall or region boundary.

Objects with unusable positions are skipped and reported. A target layer with
no usable positions causes an error. Repeating the repair adds no duplicates.

Without `--output`, the tool writes `*.object_place_edges.json` beside the
source. Use `--dry-run` to preview changes or `--force` to replace an output.

After updating a configured source scene, reload it into Neo4j and
[prepare a new dataset version](dataset_generation.md#prepare-a-dataset-version).
