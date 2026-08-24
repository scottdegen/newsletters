from corpus.fuse import reciprocal_rank_fusion


def test_rrf_boosts_items_appearing_in_multiple_lists():
    dense = [1, 2, 3]
    sparse = [2, 1, 4]
    fused = reciprocal_rank_fusion([dense, sparse])
    fused_ids = [pair[0] for pair in fused]
    # 1 and 2 appear in both lists at good ranks; 3 and 4 appear in only one.
    assert fused_ids[0] in (1, 2)
    assert fused_ids[1] in (1, 2)
    assert set(fused_ids[:2]) == {1, 2}


def test_rrf_single_list_preserves_rank_order():
    fused = reciprocal_rank_fusion([[10, 20, 30]])
    assert [pair[0] for pair in fused] == [10, 20, 30]


def test_rrf_ties_break_by_id_ascending():
    # Two disjoint lists, no overlap -> every id gets the same score at the
    # same rank position (1/(60+1) for both list[0] entries etc).
    fused = reciprocal_rank_fusion([[5], [3]])
    assert fused[0][1] == fused[1][1]  # identical scores
    assert [pair[0] for pair in fused] == [3, 5]  # tie broken ascending


def test_rrf_is_deterministic():
    lists = [[3, 1, 2], [2, 3, 1]]
    assert reciprocal_rank_fusion(lists) == reciprocal_rank_fusion(lists)


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
