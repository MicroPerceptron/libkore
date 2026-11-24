//! Checked multidimensional grids, field layouts, neighborhoods, and tiles.
//!
//! These types describe geometric indexing only. They neither allocate nor
//! replace a tensor backend's storage layout; callers supply the backing
//! slices and decide how tiles are scheduled.

use axmath::floor_f32;

/// Validation or indexing failure for grid and field views.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridError {
    EmptyAxis(usize),
    InvalidOrigin(usize),
    InvalidSpacing(usize),
    Overflow,
    OutOfBounds,
    BackingTooSmall { required: usize, actual: usize },
}

/// A regular `D`-dimensional sample lattice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSpec<const D: usize> {
    pub extent: [u32; D],
    pub origin: [f32; D],
    pub spacing: [f32; D],
}

impl<const D: usize> GridSpec<D> {
    pub fn new(extent: [u32; D], origin: [f32; D], spacing: [f32; D]) -> Result<Self, GridError> {
        for axis in 0..D {
            if extent[axis] == 0 {
                return Err(GridError::EmptyAxis(axis));
            }
            if !origin[axis].is_finite() {
                return Err(GridError::InvalidOrigin(axis));
            }
            if !spacing[axis].is_finite() || spacing[axis] == 0.0 {
                return Err(GridError::InvalidSpacing(axis));
            }
        }
        let grid = Self {
            extent,
            origin,
            spacing,
        };
        grid.element_count()?;
        Ok(grid)
    }

    pub fn element_count(&self) -> Result<usize, GridError> {
        self.extent.iter().try_fold(1_usize, |count, &extent| {
            count
                .checked_mul(extent as usize)
                .ok_or(GridError::Overflow)
        })
    }

    #[inline]
    pub fn contains(&self, index: [u32; D]) -> bool {
        (0..D).all(|axis| index[axis] < self.extent[axis])
    }

    /// Row-major linearization with the last axis contiguous.
    pub fn linearize(&self, index: [u32; D]) -> Result<usize, GridError> {
        if !self.contains(index) {
            return Err(GridError::OutOfBounds);
        }
        let mut linear = 0_usize;
        for (axis, &coordinate) in index.iter().enumerate() {
            linear = linear
                .checked_mul(self.extent[axis] as usize)
                .and_then(|value| value.checked_add(coordinate as usize))
                .ok_or(GridError::Overflow)?;
        }
        Ok(linear)
    }

    /// Inverse of [`linearize`](Self::linearize).
    pub fn delinearize(&self, mut linear: usize) -> Result<[u32; D], GridError> {
        if linear >= self.element_count()? {
            return Err(GridError::OutOfBounds);
        }
        let mut index = [0; D];
        for axis in (0..D).rev() {
            let extent = self.extent[axis] as usize;
            index[axis] = (linear % extent) as u32;
            linear /= extent;
        }
        Ok(index)
    }

    /// World-space position of a lattice sample.
    #[inline]
    pub fn sample_position(&self, index: [u32; D]) -> Result<[f32; D], GridError> {
        if !self.contains(index) {
            return Err(GridError::OutOfBounds);
        }
        Ok(core::array::from_fn(|axis| {
            self.origin[axis] + index[axis] as f32 * self.spacing[axis]
        }))
    }

    /// Integer cell containing a world-space point, before boundary handling.
    pub fn point_cell(&self, point: [f32; D]) -> Result<[i64; D], GridError> {
        let mut cell = [0_i64; D];
        for axis in 0..D {
            if !point[axis].is_finite() {
                return Err(GridError::OutOfBounds);
            }
            cell[axis] = floor_f32((point[axis] - self.origin[axis]) / self.spacing[axis]) as i64;
        }
        Ok(cell)
    }
}

/// Geometric component layout expressed in element strides.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldLayout<const D: usize> {
    pub strides: [usize; D],
    pub component_stride: usize,
}

impl<const D: usize> FieldLayout<D> {
    /// Structure-of-arrays layout: all samples of one component are contiguous.
    pub fn soa<const C: usize>(grid: &GridSpec<D>) -> Result<Self, GridError> {
        if C == 0 {
            return Err(GridError::Overflow);
        }
        Ok(Self {
            strides: row_major_strides(grid.extent, 1)?,
            component_stride: grid.element_count()?,
        })
    }

    /// Array-of-structures layout: components of one sample are contiguous.
    pub fn aos<const C: usize>(grid: &GridSpec<D>) -> Result<Self, GridError> {
        if C == 0 {
            return Err(GridError::Overflow);
        }
        Ok(Self {
            strides: row_major_strides(grid.extent, C)?,
            component_stride: 1,
        })
    }

    pub fn required_len<const C: usize>(&self, grid: &GridSpec<D>) -> Result<usize, GridError> {
        if C == 0 {
            return Ok(0);
        }
        let mut maximum = (C - 1)
            .checked_mul(self.component_stride)
            .ok_or(GridError::Overflow)?;
        for axis in 0..D {
            maximum = maximum
                .checked_add(
                    (grid.extent[axis] as usize - 1)
                        .checked_mul(self.strides[axis])
                        .ok_or(GridError::Overflow)?,
                )
                .ok_or(GridError::Overflow)?;
        }
        maximum.checked_add(1).ok_or(GridError::Overflow)
    }

    pub fn offset<const C: usize>(
        &self,
        grid: &GridSpec<D>,
        index: [u32; D],
        component: usize,
    ) -> Result<usize, GridError> {
        if component >= C || !grid.contains(index) {
            return Err(GridError::OutOfBounds);
        }
        let mut offset = component
            .checked_mul(self.component_stride)
            .ok_or(GridError::Overflow)?;
        for (axis, &coordinate) in index.iter().enumerate() {
            offset = offset
                .checked_add(
                    (coordinate as usize)
                        .checked_mul(self.strides[axis])
                        .ok_or(GridError::Overflow)?,
                )
                .ok_or(GridError::Overflow)?;
        }
        Ok(offset)
    }
}

fn row_major_strides<const D: usize>(
    extent: [u32; D],
    innermost: usize,
) -> Result<[usize; D], GridError> {
    let mut strides = [0; D];
    let mut stride = innermost;
    for axis in (0..D).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(extent[axis] as usize)
            .ok_or(GridError::Overflow)?;
    }
    Ok(strides)
}

/// Checked immutable view of a `C`-component regular field.
pub struct FieldView<'a, T, const D: usize, const C: usize> {
    data: &'a [T],
    pub grid: GridSpec<D>,
    pub layout: FieldLayout<D>,
}

impl<'a, T, const D: usize, const C: usize> FieldView<'a, T, D, C> {
    pub fn new(
        data: &'a [T],
        grid: GridSpec<D>,
        layout: FieldLayout<D>,
    ) -> Result<Self, GridError> {
        if C == 0 {
            return Err(GridError::Overflow);
        }
        let required = layout.required_len::<C>(&grid)?;
        if data.len() < required {
            return Err(GridError::BackingTooSmall {
                required,
                actual: data.len(),
            });
        }
        Ok(Self { data, grid, layout })
    }

    #[inline]
    pub fn get(&self, index: [u32; D], component: usize) -> Result<&'a T, GridError> {
        let offset = self.layout.offset::<C>(&self.grid, index, component)?;
        self.data.get(offset).ok_or(GridError::OutOfBounds)
    }
}

/// Boundary behavior for sampled-field neighborhoods.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary<T> {
    Clamp,
    Wrap,
    Mirror,
    Constant(T),
}

/// Center and immediate axis neighbors for a pointwise stencil.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisNeighbors<T, const D: usize, const C: usize> {
    pub center: [T; C],
    pub minus: [[T; C]; D],
    pub plus: [[T; C]; D],
}

impl<'a, T: Copy, const D: usize, const C: usize> FieldView<'a, T, D, C> {
    pub fn axis_neighbors(
        &self,
        index: [u32; D],
        boundary: Boundary<T>,
    ) -> Result<AxisNeighbors<T, D, C>, GridError> {
        if !self.grid.contains(index) {
            return Err(GridError::OutOfBounds);
        }
        let center = self.sample_components(index.map(i64::from), boundary)?;
        let mut minus = [center; D];
        let mut plus = [center; D];
        for axis in 0..D {
            let mut low = index.map(i64::from);
            let mut high = low;
            low[axis] -= 1;
            high[axis] += 1;
            minus[axis] = self.sample_components(low, boundary)?;
            plus[axis] = self.sample_components(high, boundary)?;
        }
        Ok(AxisNeighbors {
            center,
            minus,
            plus,
        })
    }

    fn sample_components(
        &self,
        index: [i64; D],
        boundary: Boundary<T>,
    ) -> Result<[T; C], GridError> {
        let mut resolved = [0_u32; D];
        for axis in 0..D {
            let Some(value) = resolve_axis(index[axis], self.grid.extent[axis], boundary) else {
                let Boundary::Constant(value) = boundary else {
                    return Err(GridError::OutOfBounds);
                };
                return Ok([value; C]);
            };
            resolved[axis] = value;
        }
        let mut values = [*self.get(resolved, 0)?; C];
        for (component, value) in values.iter_mut().enumerate().skip(1) {
            *value = *self.get(resolved, component)?;
        }
        Ok(values)
    }
}

fn resolve_axis<T: Copy>(index: i64, extent: u32, boundary: Boundary<T>) -> Option<u32> {
    let extent = i64::from(extent);
    if (0..extent).contains(&index) {
        return Some(index as u32);
    }
    match boundary {
        Boundary::Constant(_) => None,
        Boundary::Clamp => Some(index.clamp(0, extent - 1) as u32),
        Boundary::Wrap => Some(index.rem_euclid(extent) as u32),
        Boundary::Mirror if extent == 1 => Some(0),
        Boundary::Mirror => {
            let period = 2 * (extent - 1);
            let folded = index.rem_euclid(period);
            Some(if folded < extent {
                folded as u32
            } else {
                (period - folded) as u32
            })
        }
    }
}

/// A domain tile with asymmetric per-axis halo widths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileSpec<const D: usize> {
    pub origin: [u32; D],
    pub extent: [u32; D],
    pub halo: [[u16; 2]; D],
}

impl<const D: usize> TileSpec<D> {
    /// Tile bounds including halo, clipped to a domain.
    pub fn expanded_bounds(&self, domain: [u32; D]) -> Result<([u32; D], [u32; D]), GridError> {
        let mut start = [0; D];
        let mut end = [0; D];
        for axis in 0..D {
            if self.origin[axis] > domain[axis]
                || self.extent[axis] > domain[axis] - self.origin[axis]
            {
                return Err(GridError::OutOfBounds);
            }
            start[axis] = self.origin[axis].saturating_sub(u32::from(self.halo[axis][0]));
            end[axis] = self.origin[axis]
                .checked_add(self.extent[axis])
                .and_then(|value| value.checked_add(u32::from(self.halo[axis][1])))
                .ok_or(GridError::Overflow)?
                .min(domain[axis]);
        }
        Ok((start, end))
    }
}

/// Regular tiling of a multidimensional domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileGrid<const D: usize> {
    pub domain: [u32; D],
    pub tile: [u32; D],
}

impl<const D: usize> TileGrid<D> {
    pub fn new(domain: [u32; D], tile: [u32; D]) -> Result<Self, GridError> {
        for axis in 0..D {
            if domain[axis] == 0 || tile[axis] == 0 {
                return Err(GridError::EmptyAxis(axis));
            }
        }
        Ok(Self { domain, tile })
    }

    pub fn counts(&self) -> [u32; D] {
        core::array::from_fn(|axis| self.domain[axis].div_ceil(self.tile[axis]))
    }

    pub fn tile_count(&self) -> Result<usize, GridError> {
        self.counts().iter().try_fold(1_usize, |count, &value| {
            count.checked_mul(value as usize).ok_or(GridError::Overflow)
        })
    }

    pub fn tile_at(&self, coordinate: [u32; D]) -> Result<TileSpec<D>, GridError> {
        let counts = self.counts();
        if (0..D).any(|axis| coordinate[axis] >= counts[axis]) {
            return Err(GridError::OutOfBounds);
        }
        let origin = core::array::from_fn(|axis| coordinate[axis] * self.tile[axis]);
        let extent =
            core::array::from_fn(|axis| self.tile[axis].min(self.domain[axis] - origin[axis]));
        Ok(TileSpec {
            origin,
            extent,
            halo: [[0, 0]; D],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_indexing_roundtrips() {
        let grid = GridSpec::new([3, 4, 5], [0.0; 3], [1.0; 3]).unwrap();
        for linear in 0..grid.element_count().unwrap() {
            assert_eq!(
                grid.linearize(grid.delinearize(linear).unwrap()).unwrap(),
                linear
            );
        }
        assert_eq!(grid.sample_position([2, 3, 4]).unwrap(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn aos_and_soa_views_address_components() {
        let grid = GridSpec::new([2, 3], [0.0; 2], [1.0; 2]).unwrap();
        let aos_data: [u32; 12] = core::array::from_fn(|index| index as u32);
        let aos = FieldView::<_, 2, 2>::new(&aos_data, grid, FieldLayout::aos::<2>(&grid).unwrap())
            .unwrap();
        assert_eq!(*aos.get([1, 2], 0).unwrap(), 10);
        assert_eq!(*aos.get([1, 2], 1).unwrap(), 11);

        let soa_data: [u32; 12] = core::array::from_fn(|index| index as u32);
        let soa = FieldView::<_, 2, 2>::new(&soa_data, grid, FieldLayout::soa::<2>(&grid).unwrap())
            .unwrap();
        assert_eq!(*soa.get([1, 2], 0).unwrap(), 5);
        assert_eq!(*soa.get([1, 2], 1).unwrap(), 11);
    }

    #[test]
    fn neighborhoods_apply_each_boundary_policy() {
        let grid = GridSpec::new([3], [0.0], [1.0]).unwrap();
        let data = [10, 20, 30];
        let view =
            FieldView::<_, 1, 1>::new(&data, grid, FieldLayout::soa::<1>(&grid).unwrap()).unwrap();
        assert_eq!(
            view.axis_neighbors([0], Boundary::Clamp).unwrap().minus[0],
            [10]
        );
        assert_eq!(
            view.axis_neighbors([0], Boundary::Wrap).unwrap().minus[0],
            [30]
        );
        assert_eq!(
            view.axis_neighbors([0], Boundary::Mirror).unwrap().minus[0],
            [20]
        );
        assert_eq!(
            view.axis_neighbors([0], Boundary::Constant(-1))
                .unwrap()
                .minus[0],
            [-1]
        );
    }

    #[test]
    fn partial_tiles_and_halos_clip_to_domain() {
        let tiles = TileGrid::new([10, 7], [4, 4]).unwrap();
        assert_eq!(tiles.counts(), [3, 2]);
        assert_eq!(tiles.tile_count().unwrap(), 6);
        let mut edge = tiles.tile_at([2, 1]).unwrap();
        assert_eq!(edge.origin, [8, 4]);
        assert_eq!(edge.extent, [2, 3]);
        edge.halo = [[2, 3], [5, 1]];
        assert_eq!(
            edge.expanded_bounds(tiles.domain).unwrap(),
            ([6, 0], [10, 7])
        );
    }

    #[test]
    fn views_validate_reachable_storage() {
        let grid = GridSpec::new([4, 4], [0.0; 2], [1.0; 2]).unwrap();
        let error =
            FieldView::<_, 2, 3>::new(&[0_u8; 47], grid, FieldLayout::soa::<3>(&grid).unwrap())
                .err()
                .unwrap();
        assert_eq!(
            error,
            GridError::BackingTooSmall {
                required: 48,
                actual: 47
            }
        );
    }
}
