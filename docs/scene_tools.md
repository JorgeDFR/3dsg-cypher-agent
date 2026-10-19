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
accepts either a listed label (including names such as `conference room` and
`living room`) or its number. Types cover indoor rooms and outdoor regions, such
as `garden`, `courtyard`, `sidewalk`, `park`, and `parking lot`.

Only the listed canonical names are accepted.
Input ignores case and extra whitespace.
Existing room label IDs are retained.
Enter keeps a current canonical type.

### Output behavior

| Option | Behavior |
| --- | --- |
| No `--output` | Preserve the source and write `*.room_labels.json` beside it |
| `--output PATH` | Write the updated scene to `PATH` |
| `--figure-output PATH` | Save the room visualization |
| `--no-show` | Run without opening the graphical window |
| `--list-labels` | Print the controlled room vocabulary |
| `--force` | Allow replacing an existing output, including the source |

## Repair missing object-to-place connections

Connect objects to their nearest mesh place:

```bash
docker/stack.sh main python scripts/graph/fix_3dsg_object_place_edges.py \
  data/scenes/uhumans2_office.json \
  --target-layer mesh-place \
  --output data/scenes/mit_courtyard_ade20k_full.connected.json
```

Use `--target-layer place` to select the place layer instead. The script respects
the scene's named layer and partition mapping. In the MIT courtyard scene,
`PLACES` and `MESH_PLACES` alias the same layer, so both choices select the same
nodes.

For each object without an edge to the chosen layer, it adds one standard Spark
DSG edge from the nearest place to the object. Nearest means Euclidean distance
between node positions in XYZ; equal distances are resolved by node ID. Existing
connections in either orientation are kept, including connections to other
layers. Repeating the repair adds no duplicate edges. Objects with missing or
nonfinite positions are skipped and reported; a target layer with no usable
positions causes an error.

Without `--output`, the script writes `*.object_place_edges.json` beside the
source. Use `--dry-run` to report changes without writing or `--force` to replace
an existing output. Nodes, metadata, mesh data (when present), and existing edges
are preserved. This is a geometric approximation of containment; nearby nodes
can lie across a wall or region boundary.

After changing the scene used by `configs/data.yaml`, reload it into Neo4j and
start a new dataset version. See
[Prepare a dataset version](dataset_generation.md#prepare-a-dataset-version).
