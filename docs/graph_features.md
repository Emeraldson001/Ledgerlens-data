# Graph Features: Motif Census

## Overview

Once [Louvain community detection](#g-community) (`detection/community_detector.py`) partitions the wallet graph into suspected wash-trading [rings](#g-ring), the [motif](#g-motif) census (`detection/motif_census.py`) characterises the *internal structure* of each [community](#g-community) by counting k-node subgraph patterns (motifs). The resulting structural fingerprints distinguish coordinated wash rings (which favour dense triangles and reciprocal cycles) from organic market-maker networks (which tend toward hub-and-spoke star topologies with low [reciprocity](#g-reciprocity)).

Entry point:

```python
from detection.community_detector import enrich_communities_with_motifs

motif_features = enrich_communities_with_motifs(graph, community_map)
# Returns: {community_id: {"triangle_density": ..., "star_ratio": ..., ...}}
```

---

## Glossary

A quick reference for the graph-theory vocabulary used in this document and in
the detection modules it describes. Each entry gives a one- or two-sentence
plain-English definition and a pointer to the exact function that computes or
defines the concept. Terms are cross-linked from their first use above.

### <a id="g-funding-edge"></a>Funding edge

A directed edge pointing from the account that funded another account to the
account it funded ("funder → funded"). These edges are the backbone of the
wallet graph; co-trading wallets add further edges on top.
*See `build_funding_graph()` in [`detection/wallet_graph.py`](../detection/wallet_graph.py) (edges tagged `edge_type="funding"`).*

### <a id="g-ancestor-traversal"></a>Ancestor traversal

Walking [funding edges](#g-funding-edge) *backwards* from a wallet to collect the
set of accounts that funded it, then funded those, and so on, up to a fixed hop
limit. Two wallets whose ancestor sets overlap heavily probably share a common
paymaster — a classic sock-puppet / [ring](#g-ring) signal.
*See `multi_hop_ancestors()` in [`detection/wallet_graph.py`](../detection/wallet_graph.py), consumed by `_funding_source_similarity()`.*

### <a id="g-community"></a>Community

A group of wallets that are far more densely connected to each other than to the
rest of the graph. LedgerLens finds communities with the **Louvain** algorithm,
which greedily maximises modularity; a fixed random seed keeps the partition
deterministic in CI.
*See `detect_communities()` in [`detection/community_detector.py`](../detection/community_detector.py).*

### <a id="g-ring"></a>Ring

A [community](#g-community) large enough (`>= min_ring_size`) to be treated as a
suspected coordinated wash-trading group. Rings get a stable content-addressed id
derived from their sorted member set so the same group keeps the same label
across runs.
*See `detect_wash_trading_rings()` and `ring_id_for_members()` in [`detection/wallet_graph.py`](../detection/wallet_graph.py).*

### <a id="g-internal-edge-density"></a>Internal edge density

For one [ring](#g-ring), the fraction of the edges that *could* exist between its
members that *actually* exist: `internal_edges / (k * (k - 1) / 2)` for `k`
members. `1.0` means every member trades with every other member (a clique);
values near `0` mean the members are barely connected to each other.
*See `ring_statistics()` in [`detection/wallet_graph.py`](../detection/wallet_graph.py) (key `internal_edge_density`).*

### <a id="g-motif"></a>Motif

A small connected subgraph pattern (here, 3 or 4 nodes) — for example a triangle,
an open wedge, or a 4-cycle. Counting how often each pattern occurs inside a
[community](#g-community) gives a size-independent structural fingerprint.
*See `compute_motif_census()` in [`detection/motif_census.py`](../detection/motif_census.py).*

### <a id="g-reciprocity"></a>Reciprocity

The fraction of directed edges `(u, v)` for which the reverse edge `(v, u)` also
exists. Wash rings show high reciprocity (round-trip flows); organic trading
produces more one-way paths. Undirected graphs are defined to have reciprocity
`1.0`. Full formula in [Reciprocity](#reciprocity) below.
*Computed inside `compute_motif_census()` in [`detection/motif_census.py`](../detection/motif_census.py).*

---

## 3-Node Motif Taxonomy

| Motif | Pattern | Edges | Detection method |
|-------|---------|-------|-----------------|
| **Triangle** (K₃) | All three nodes mutually connected | 3 | A³ matrix trace |
| **Open wedge / Star** (P₃) | One centre node connected to two leaves; leaves not connected | 2 | Degree-sum formula |

### Triangle Density

```
triangle_density = triangle_count / C(n, 3)
```

where `n` is the community size and `C(n, 3) = n(n−1)(n−2)/6` is the maximum possible number of triangles.  A value of 1.0 means every triple of nodes forms a triangle (i.e., the community is a clique).

**Efficient computation — A³ trace method**

Rather than enumerating every node triple (O(n³)), triangle count is computed via:

```
triangles = trace(A³) / 6
```

where A is the adjacency matrix of the undirected community subgraph. This reduces to two matrix multiplications (O(n^{2.37…}) with optimised BLAS, O(n³) worst-case) followed by a diagonal sum, which is faster in practice because NumPy delegates to LAPACK/OpenBLAS.

### Star Ratio

```
total_wedges    = Σ_v C(deg_v, 2)
open_wedges     = total_wedges  −  3 × triangle_count
star_ratio      = open_wedges / (open_wedges + triangle_count)
```

Open wedges (P₃ patterns) are wedges whose two endpoints are *not* directly connected; each triangle closes exactly 3 wedges, hence the correction. A value near 1.0 identifies hub-and-spoke structures (market makers or relay accounts); a value near 0.0 signals dense cliques (wash rings).

---

## 4-Node Motif Taxonomy

| Motif | Pattern | Wash-ring signal |
|-------|---------|-----------------|
| **4-cycle** (C₄) | Square: a–b–c–d–a | High — closed loops with no shared centre |
| **4-path** (P₄) | Linear chain: a–b–c–d | Neutral — common in liquidity routing |
| **Star** (K₁,₃) | One hub, three leaves | Low — hub-spoke topology |
| **Diamond** (K₄ − edge) | 4 nodes, 5 edges, missing one edge | High — near-clique |
| **Complete** (K₄) | 6 edges, all connected | Very high — perfect clique |

### cycle\_4\_count

The number of distinct induced 4-cycles is computed from the A⁴ trace formula:

```
trace(A⁴) = 2m  +  2·Σ_v d_v(d_v−1)  +  8·C4
```

Rearranging:

```
C4 = ( trace(A⁴) − 2m − 2·Σ_v d_v(d_v−1) ) / 8
```

where `m = |E|` and `d_v` is the degree of node v.

`trace(A⁴)` is computed as `Σ_{i,j} (A²)_{ij}²` — a single element-wise square and sum after one matrix multiply, avoiding an explicit fourth-power matrix.

The raw integer count is stored in `MotifCensusResult.cycle_4_count`.  When integrated into the community feature vector via `enrich_communities_with_motifs`, it is normalised by community size:

```
cycle_4_per_node = cycle_4_count / n
```

---

## Reciprocity

```
reciprocity = |{(u,v) ∈ E : (v,u) ∈ E}| / |E|
```

Reciprocity measures the fraction of directed edges that have a matching reverse edge.  Wash rings tend to exhibit high reciprocity (coordinated round-trip flows), whereas organic trading creates more directed, one-way paths.

For undirected graphs, reciprocity is defined as 1.0 by convention.

---

## Feature Normalisation

All features are normalised so they are comparable across communities of different sizes:

| Feature | Normalisation |
|---------|--------------|
| `triangle_density` | Divided by `C(n, 3)` — ranges 0–1 |
| `star_ratio` | Ratio of open-wedge to total 3-node motifs — ranges 0–1 |
| `cycle_4_per_node` | `cycle_4_count / n` — rate per node |
| `reciprocity` | Already a fraction — ranges 0–1 |

---

## Timeout and Sampling Strategy

### Large community sampling

Communities with more than **500 nodes** are replaced by a 500-node random induced subgraph before any matrix operations are performed.  The sampled subgraph uses a fixed NumPy seed (42) for reproducibility.  The `was_sampled` flag in `MotifCensusResult` indicates when this occurred.

### Timeout

The motif census is time-bounded by `MOTIF_CENSUS_TIMEOUT_SECONDS` (default **5 seconds**, configurable via environment variable).  The deadline is checked between each computation phase:

1. Triangle counting (A³ method)
2. Star/open-wedge counting
3. 4-cycle counting (A⁴ method)
4. Reciprocity computation

If the deadline is exceeded at any checkpoint, computation halts immediately and partial results are returned with `census_truncated = True`.  Downstream consumers should treat truncated results as lower-confidence signals and may choose to exclude them from model input.

```python
result = compute_motif_census(subgraph, known_nodes, timeout_seconds=2.0)
if result.census_truncated:
    logger.warning("Motif census truncated for community %s", cid)
```

---

## Security: Subgraph Validation

`compute_motif_census` validates that every node in the supplied subgraph is present in the `known_nodes` set (the node set of the parent wallet graph).  Subgraphs referencing external wallet addresses are rejected with a `ValueError` before any computation begins, preventing injection of synthetic nodes that could skew structural features.

```python
# Raises ValueError: "Subgraph contains N node(s) not in the known graph: ..."
compute_motif_census(subgraph_with_external_node, known_nodes)
```

---

## Integration Example

```python
from detection.community_detector import detect_communities, enrich_communities_with_motifs

community_map = detect_communities(graph)
motif_features = enrich_communities_with_motifs(graph, community_map, timeout_seconds=5.0)

for cid, features in motif_features.items():
    print(
        f"Community {cid}: "
        f"triangle_density={features['triangle_density']:.3f}, "
        f"star_ratio={features['star_ratio']:.3f}, "
        f"cycle_4_per_node={features['cycle_4_per_node']:.4f}, "
        f"reciprocity={features['reciprocity']:.3f}"
        + (" [TRUNCATED]" if features["census_truncated"] else "")
    )
```
