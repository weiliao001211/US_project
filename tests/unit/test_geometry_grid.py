def test_index_coord_inverse(tiny_grid):
    for ix in (0, tiny_grid.nx // 2, tiny_grid.nx - 1):
        for iy in (0, tiny_grid.ny // 2, tiny_grid.ny - 1):
            x, y = tiny_grid.index2coord(ix, iy)
            ix2, iy2 = tiny_grid.coord2index(x, y)
            assert (ix, iy) == (ix2, iy2)


def test_ring_masks_shape_and_transpose(tiny_grid, ring8):
    m_rx = ring8.get_rx_mask(tiny_grid)  # (nx, ny)
    m_tx = ring8.get_tx_mask(tiny_grid)  # (nx, ny)
    assert m_rx.shape == (tiny_grid.nx, tiny_grid.ny)
    assert m_tx.shape == (tiny_grid.nx, tiny_grid.ny)
    # WaveOperator relies on .T to become (ny,nx); lock this in:
    assert m_rx.T.shape == (tiny_grid.ny, tiny_grid.nx)
