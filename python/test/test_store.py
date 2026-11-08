import numpy as np
import torch

import tguf


def test_tgstore_from_memory(get_data):
    num_edges = 20
    batch = tguf.Batch(
        src=get_data("numpy", np.arange(num_edges), "int64"),
        dst=get_data("numpy", np.arange(num_edges) + 1, "int64"),
        time=get_data("numpy", np.arange(num_edges) * 10, "int64"),
        msg=get_data("numpy", np.random.randn(num_edges, 8), "float32"),
    )
    store = tguf.TGStore.from_memory(edges=batch, val_start=15, test_start=18)

    assert store.edge_count == num_edges
    assert store.node_count == num_edges + 1
    assert store.label_count == 0
    assert store.msg_dim == 8
    assert store.label_dim == 0
    assert store.train_split.end == 15
    assert store.val_split.size == 3
    assert store.test_split.start == 18

    b = store.get_batch(0, 5, tguf.NegStrategy.None_)
    assert torch.is_tensor(b.src)
    assert torch.is_tensor(b.dst)
    assert torch.is_tensor(b.time)
    assert torch.is_tensor(b.msg)
    assert b.neg_dst is None


def test_tgstore_from_tguf(schema, get_data):
    builder = tguf.TGUFBuilder(schema)
    num_edges = 20
    batch = tguf.Batch(
        src=get_data("numpy", np.arange(num_edges), "int64"),
        dst=get_data("numpy", np.arange(num_edges) + 1, "int64"),
        time=get_data("numpy", np.arange(num_edges) * 10, "int64"),
        msg=get_data("numpy", np.random.randn(num_edges, 8), "float32"),
    )
    builder.append_edges(batch)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path, val_start=15, test_start=18)

    assert store.edge_count == num_edges
    assert store.node_count == num_edges + 1
    assert store.label_count == 0
    assert store.msg_dim == 8
    assert store.label_dim == 0
    assert store.train_split.end == 15
    assert store.val_split.size == 3
    assert store.test_split.start == 18

    b = store.get_batch(0, 5, tguf.NegStrategy.None_)
    assert torch.is_tensor(b.src)
    assert torch.is_tensor(b.dst)
    assert torch.is_tensor(b.time)
    assert torch.is_tensor(b.msg)
    assert b.neg_dst is None


def test_tgstore_node_features(get_data):
    num_nodes = 5
    feat_dim = 4
    feats = np.random.randn(num_nodes, feat_dim).astype(np.float32)

    batch = tguf.Batch(
        src=get_data("numpy", [0], "int64"),
        dst=get_data("numpy", [1], "int64"),
        time=get_data("numpy", [0], "int64"),
        msg=get_data("numpy", np.zeros((1, 8)), "float32"),
    )

    store = tguf.TGStore.from_memory(
        edges=batch,
        node_feats=torch.from_numpy(feats),
        label_n_id=None,
        label_time=None,
        label_target=None,
        val_start=None,
        test_start=None,
    )

    query_nodes = torch.tensor([4, 0, 2], dtype=torch.long)
    out = store.gather_node_feats(query_nodes)

    torch.testing.assert_close(out, torch.from_numpy(feats[[4, 0, 2]]))


def test_tgstore_split_logic(schema):
    schema.val_start = 8
    schema.test_start = 9
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=np.zeros(10, dtype=np.int64),
        dst=np.zeros(10, dtype=np.int64),
        time=np.arange(10, dtype=np.int64),
        msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
    )
    builder.append_edges(batch)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    assert store.train_split.start == 0
    assert store.train_split.end == 8
    assert store.val_split.start == 8
    assert store.val_split.size == 1  # 9 - 8
    assert store.test_split.start == 9
    assert store.test_split.size == 1  # 10 - 9


def test_tgstore_vectorized_gathers(schema):
    times = np.array([100, 200, 300, 400], dtype=np.int64)
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=np.zeros(4, dtype=np.int64),
        dst=np.zeros(4, dtype=np.int64),
        time=times,
        msg=np.zeros((4, schema.msg_dim), dtype=np.float32),
    )
    builder.append_edges(batch)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    indices = torch.tensor([3, 0, 1], dtype=torch.long)
    t_out = store.gather_timestamps(indices)

    assert torch.equal(t_out, torch.tensor([400, 100, 200], dtype=torch.long))


def test_tgstore_label_event_and_cutoff(schema):
    builder = tguf.TGUFBuilder(schema)
    # Add 10 edges at times 0, 10, 20...
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10),
            dst=np.zeros(10),
            time=np.arange(10) * 10,
            msg=np.zeros((10, schema.msg_dim)),
        )
    )
    # Add a label at time 45 (should cutoff at edge_id 4, which is time 40)
    builder.append_labels(
        n_id=np.array([1]), time=np.array([45]), target=np.array([[1]])
    )
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    cutoff = store.get_edge_cutoff_for_label_event(0)
    assert cutoff == 5  # 5 edges (0, 10, 20, 30, 40) are <= time 45

    label = store.get_label_event(0)
    assert label.n_id[0] == 1
    assert label.target.shape == (1, schema.label_dim)


def test_tensor_independence(schema):
    num_edges = 10
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=np.zeros(num_edges, dtype=np.int64),
        dst=np.zeros(num_edges, dtype=np.int64),
        time=np.arange(num_edges, dtype=np.int64) * 10,
        msg=np.zeros((num_edges, schema.msg_dim), dtype=np.float32),
    )
    builder.append_edges(batch)
    builder.finalize()
    store = tguf.TGStore.from_tguf(schema.path)

    indices = torch.tensor([0, 1, 2], dtype=torch.long)
    t_out = store.gather_timestamps(indices)

    # Delete the store (triggering C++ destructors)
    del store

    # Try to access the data. If the capsule worked, this remains valid.
    assert t_out[0] == 0
    assert t_out.sum() == 30  # (0 + 10 + 20)


def test_empty_gather(schema):
    num_edges = 10
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=np.zeros(num_edges, dtype=np.int64),
        dst=np.zeros(num_edges, dtype=np.int64),
        time=np.arange(num_edges, dtype=np.int64) * 10,
        msg=np.zeros((num_edges, schema.msg_dim), dtype=np.float32),
    )
    builder.append_edges(batch)
    builder.finalize()
    store = tguf.TGStore.from_tguf(schema.path)
    empty_indices = torch.tensor([], dtype=torch.long)
    t_out = store.gather_timestamps(empty_indices)

    assert t_out.shape[0] == 0
    assert torch.is_tensor(t_out)


def test_tgstore_label_event_table(schema):
    schema.label_capacity = 6
    builder = tguf.TGUFBuilder(schema)
    # Add 10 edges at times 0, 10, 20...
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10, dtype=np.int64),
            dst=np.zeros(10, dtype=np.int64),
            time=np.arange(10, dtype=np.int64) * 10,
            msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
        )
    )
    # 6 label rows in 3 label events: t=15 (x2), t=25 (x1), t=45 (x3)
    builder.append_labels(
        n_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        time=np.array([15, 15, 25, 45, 45, 45], dtype=np.int64),
        target=np.arange(6, dtype=np.float32).reshape(6, 1),
    )
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    table = store.label_event_table  # property, no parentheses
    assert isinstance(table, tguf.LabelEventTable)
    assert torch.equal(table.time, torch.tensor([15, 25, 45]))
    assert torch.equal(table.row_offset, torch.tensor([0, 2, 3, 6]))
    assert torch.equal(table.cutoff, torch.tensor([2, 3, 5]))  # edges before t
    assert table.time.dtype == torch.int64
    assert table.row_offset.dtype == torch.int64
    assert table.cutoff.dtype == torch.int64
    assert table.row_offset[-1] == store.label_count


def test_tgstore_label_event_table_matches_label_event(schema):
    schema.label_capacity = 6
    builder = tguf.TGUFBuilder(schema)
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10, dtype=np.int64),
            dst=np.zeros(10, dtype=np.int64),
            time=np.arange(10, dtype=np.int64) * 10,
            msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
        )
    )
    builder.append_labels(
        n_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        time=np.array([15, 15, 25, 45, 45, 45], dtype=np.int64),
        target=np.arange(6, dtype=np.float32).reshape(6, 1),
    )
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    # The table must agree with the per-label-event APIs
    table = store.label_event_table
    for k in range(table.time.shape[0]):
        rows = int(table.row_offset[k + 1] - table.row_offset[k])
        assert rows == store.get_label_event(k).n_id.shape[0]
        assert int(table.cutoff[k]) == store.get_edge_cutoff_for_label_event(k)


def test_tgstore_label_batch(schema):
    schema.label_capacity = 6
    builder = tguf.TGUFBuilder(schema)
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10, dtype=np.int64),
            dst=np.zeros(10, dtype=np.int64),
            time=np.arange(10, dtype=np.int64) * 10,
            msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
        )
    )
    builder.append_labels(
        n_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        time=np.array([15, 15, 25, 45, 45, 45], dtype=np.int64),
        target=np.arange(6, dtype=np.float32).reshape(6, 1),
    )
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    b = store.get_label_batch(0, store.label_count)
    assert isinstance(b, tguf.LabelBatch)
    assert torch.equal(b.n_id, torch.tensor([1, 2, 3, 4, 5, 6]))
    assert torch.equal(b.time, torch.tensor([15, 15, 25, 45, 45, 45]))
    assert torch.equal(b.target, torch.arange(6, dtype=torch.float32).reshape(6, 1))
    assert b.n_id.dtype == torch.int64
    assert b.time.dtype == torch.int64
    assert b.target.dtype == torch.float32

    # Same rows as concatenating every label event
    events = [store.get_label_event(k) for k in range(3)]
    assert torch.equal(b.n_id, torch.cat([e.n_id for e in events]))
    assert torch.equal(b.target, torch.cat([e.target for e in events]))


def test_tgstore_label_batch_range(schema):
    schema.label_capacity = 6
    builder = tguf.TGUFBuilder(schema)
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10, dtype=np.int64),
            dst=np.zeros(10, dtype=np.int64),
            time=np.arange(10, dtype=np.int64) * 10,
            msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
        )
    )
    builder.append_labels(
        n_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        time=np.array([15, 15, 25, 45, 45, 45], dtype=np.int64),
        target=np.arange(6, dtype=np.float32).reshape(6, 1),
    )
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    # Starts inside label event 0 and ends inside label event 2
    b = store.get_label_batch(1, 3)
    assert torch.equal(b.n_id, torch.tensor([2, 3, 4]))
    assert torch.equal(b.time, torch.tensor([15, 25, 45]))
    assert torch.equal(b.target, torch.tensor([[1.0], [2.0], [3.0]]))


def test_tgstore_label_batch_clamp_and_empty(schema):
    schema.label_capacity = 6
    builder = tguf.TGUFBuilder(schema)
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10, dtype=np.int64),
            dst=np.zeros(10, dtype=np.int64),
            time=np.arange(10, dtype=np.int64) * 10,
            msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
        )
    )
    builder.append_labels(
        n_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        time=np.array([15, 15, 25, 45, 45, 45], dtype=np.int64),
        target=np.arange(6, dtype=np.float32).reshape(6, 1),
    )
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    # Size past the end is clamped to the last label row
    tail = store.get_label_batch(4, 10)
    assert torch.equal(tail.n_id, torch.tensor([5, 6]))

    # Start at or past the end gives an empty batch
    for start in (6, 100):
        empty = store.get_label_batch(start, 5)
        assert empty.n_id.shape == (0,)
        assert empty.time.shape == (0,)
        assert empty.target.shape == (0, schema.label_dim)
        assert empty.n_id.dtype == torch.int64
        assert empty.target.dtype == torch.float32


def test_tgstore_labels_from_memory(get_data):
    num_edges = 10
    batch = tguf.Batch(
        src=get_data("numpy", np.zeros(num_edges), "int64"),
        dst=get_data("numpy", np.zeros(num_edges), "int64"),
        time=get_data("numpy", np.arange(num_edges) * 10, "int64"),
        msg=get_data("numpy", np.zeros((num_edges, 8)), "float32"),
    )
    store = tguf.TGStore.from_memory(
        edges=batch,
        label_n_id=get_data("numpy", [1, 2, 3, 4, 5, 6], "int64"),
        label_time=get_data("numpy", [15, 15, 25, 45, 45, 45], "int64"),
        label_target=get_data("numpy", np.arange(6).reshape(6, 1), "float32"),
    )

    table = store.label_event_table
    assert torch.equal(table.time, torch.tensor([15, 25, 45]))
    assert torch.equal(table.row_offset, torch.tensor([0, 2, 3, 6]))
    assert torch.equal(table.cutoff, torch.tensor([2, 3, 5]))

    b = store.get_label_batch(1, 3)
    assert torch.equal(b.n_id, torch.tensor([2, 3, 4]))
    assert torch.equal(b.time, torch.tensor([15, 25, 45]))
    assert torch.equal(b.target, torch.tensor([[1.0], [2.0], [3.0]]))


def test_tgstore_label_apis_without_labels(get_data):
    num_edges = 10
    batch = tguf.Batch(
        src=get_data("numpy", np.zeros(num_edges), "int64"),
        dst=get_data("numpy", np.zeros(num_edges), "int64"),
        time=get_data("numpy", np.arange(num_edges) * 10, "int64"),
        msg=get_data("numpy", np.zeros((num_edges, 8)), "float32"),
    )
    store = tguf.TGStore.from_memory(edges=batch)

    table = store.label_event_table
    assert table.time.shape == (0,)
    assert table.cutoff.shape == (0,)
    assert torch.equal(table.row_offset, torch.tensor([0]))

    b = store.get_label_batch(0, 10)
    assert b.n_id.shape == (0,)
    assert b.time.shape == (0,)
    assert b.target.shape == (0, 0)
    assert b.target.dtype == torch.float32


def test_label_tensor_independence(schema):
    schema.label_capacity = 6
    builder = tguf.TGUFBuilder(schema)
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10, dtype=np.int64),
            dst=np.zeros(10, dtype=np.int64),
            time=np.arange(10, dtype=np.int64) * 10,
            msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
        )
    )
    builder.append_labels(
        n_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        time=np.array([15, 15, 25, 45, 45, 45], dtype=np.int64),
        target=np.arange(6, dtype=np.float32).reshape(6, 1),
    )
    builder.finalize()
    store = tguf.TGStore.from_tguf(schema.path)

    b = store.get_label_batch(0, 6)
    n_id, target = b.n_id, b.target
    table = store.label_event_table
    row_offset, cutoff = table.row_offset, table.cutoff

    # Delete the returned objects. The tensors must keep their data alive.
    del b
    del table

    assert n_id.sum() == 21  # (1 + 2 + ... + 6)
    assert target.sum() == 15  # (0 + 1 + ... + 5)
    assert torch.equal(row_offset, torch.tensor([0, 2, 3, 6]))
    assert torch.equal(cutoff, torch.tensor([2, 3, 5]))


def test_label_batch_zero_copy(schema):
    schema.label_capacity = 6
    builder = tguf.TGUFBuilder(schema)
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10, dtype=np.int64),
            dst=np.zeros(10, dtype=np.int64),
            time=np.arange(10, dtype=np.int64) * 10,
            msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
        )
    )
    builder.append_labels(
        n_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        time=np.array([15, 15, 25, 45, 45, 45], dtype=np.int64),
        target=np.arange(6, dtype=np.float32).reshape(6, 1),
    )
    builder.finalize()
    store = tguf.TGStore.from_tguf(schema.path)

    # Two calls for the same range point to the same memory (no copy)
    a = store.get_label_batch(2, 3)
    b = store.get_label_batch(2, 3)
    assert a.n_id.data_ptr() == b.n_id.data_ptr()
    assert a.time.data_ptr() == b.time.data_ptr()
    assert a.target.data_ptr() == b.target.data_ptr()


def test_label_event_table_returns_copies(schema):
    schema.label_capacity = 6
    builder = tguf.TGUFBuilder(schema)
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10, dtype=np.int64),
            dst=np.zeros(10, dtype=np.int64),
            time=np.arange(10, dtype=np.int64) * 10,
            msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
        )
    )
    builder.append_labels(
        n_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        time=np.array([15, 15, 25, 45, 45, 45], dtype=np.int64),
        target=np.arange(6, dtype=np.float32).reshape(6, 1),
    )
    builder.finalize()
    store = tguf.TGStore.from_tguf(schema.path)

    # Changing the returned tensors must not change the store
    table = store.label_event_table
    table.time.fill_(-1)
    table.cutoff.fill_(-1)
    assert torch.equal(store.label_event_table.time, torch.tensor([15, 25, 45]))
    assert torch.equal(store.label_event_table.cutoff, torch.tensor([2, 3, 5]))
