"""
Pandas DataFrame graph representations and conversions.

DataFrames are common in data science workflows, with edges as rows in one DataFrame
and optionally nodes as rows in another DataFrame with metadata.

Kinds defined here
------------------
- 'edges_dataframe': pandas DataFrame with source/target columns
- 'graph_dataframes': Dict with 'edges' and optionally 'nodes' DataFrames

Conversions
-----------
- edges DataFrame <-> edgelist
- edges DataFrame <-> nodes_and_links
- graph DataFrames <-> nodes_and_links
"""

from contextlib import suppress
import numpy as np

# Try to import pandas - only register if available
_has_pandas = False
with suppress(ImportError, ModuleNotFoundError):
    import pandas as pd

    _has_pandas = True

# Only register if both pandas and cast are available
with suppress(ImportError, ModuleNotFoundError):
    from linked.cast import register_kind, register_transformation


# ============================================================================
# Kind predicates
# ============================================================================

if _has_pandas:

    def _is_edges_dataframe(obj) -> bool:
        """Check if object is an edges DataFrame (has source and target columns)."""
        if not isinstance(obj, pd.DataFrame):
            return False
        # Common column name patterns for edges
        cols = set(obj.columns)
        return (
            ('source' in cols and 'target' in cols)
            or ('src' in cols and 'dst' in cols)
            or ('from' in cols and 'to' in cols)
            or ('source' in cols and 'dest' in cols)
        )

    def _is_graph_dataframes(obj) -> bool:
        """Check if object is a dict with 'edges' DataFrame."""
        if not isinstance(obj, dict):
            return False
        if 'edges' not in obj:
            return False
        return isinstance(obj['edges'], pd.DataFrame)

    # Register kinds
    with suppress(ImportError, ModuleNotFoundError):
        register_kind('edges_dataframe', isa=_is_edges_dataframe)
        register_kind('graph_dataframes', isa=_is_graph_dataframes)


# ============================================================================
# Edges DataFrame <-> edge list conversions
# ============================================================================

if _has_pandas:

    def edges_dataframe_to_edgelist(df: pd.DataFrame, ctx: dict = None) -> np.ndarray:
        """
        Convert edges DataFrame to edge list.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with edge data
        ctx : dict, optional
            Context with optional keys:
            - 'source_col': column name for source (default: auto-detect)
            - 'target_col': column name for target (default: auto-detect)
            - 'weight_col': column name for weight (default: 'weight')
            - 'include_weights': whether to include weights (default: auto-detect)
            - 'node_id_map': dict mapping node ids to indices

        Returns
        -------
        np.ndarray
            Edge list array

        Examples
        --------
        >>> df = pd.DataFrame({'source': [0, 1], 'target': [1, 2]})
        >>> edgelist = edges_dataframe_to_edgelist(df)
        >>> edgelist.shape
        (2, 2)
        """
        ctx = ctx or {}

        # Auto-detect source column
        source_col = ctx.get('source_col')
        if source_col is None:
            for col in ['source', 'src', 'from']:
                if col in df.columns:
                    source_col = col
                    break

        # Auto-detect target column
        target_col = ctx.get('target_col')
        if target_col is None:
            for col in ['target', 'dst', 'to', 'dest']:
                if col in df.columns:
                    target_col = col
                    break

        if source_col is None or target_col is None:
            raise ValueError(
                f"Could not detect source/target columns in DataFrame. Columns: {list(df.columns)}"
            )

        weight_col = ctx.get('weight_col', 'weight')
        include_weights = ctx.get('include_weights', None)
        node_id_map = ctx.get('node_id_map', None)

        # Auto-detect weights
        if include_weights is None:
            include_weights = weight_col in df.columns

        # Build node to index mapping if needed
        if node_id_map is None:
            # Get unique nodes
            all_nodes = pd.concat([df[source_col], df[target_col]]).unique()
            node_id_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Convert edges
        edges = []
        for _, row in df.iterrows():
            src = row[source_col]
            tgt = row[target_col]

            src_idx = node_id_map.get(src, src)
            tgt_idx = node_id_map.get(tgt, tgt)

            if include_weights and weight_col in df.columns:
                weight = row[weight_col]
                edges.append([src_idx, tgt_idx, weight])
            else:
                edges.append([src_idx, tgt_idx])

        return np.array(edges) if edges else np.empty((0, 3 if include_weights else 2))

    def edgelist_to_edges_dataframe(
        edgelist: np.ndarray, ctx: dict = None
    ) -> pd.DataFrame:
        """
        Convert edge list to edges DataFrame.

        Parameters
        ----------
        edgelist : np.ndarray
            Edge list array
        ctx : dict, optional
            Context with optional keys:
            - 'source_col': column name for source (default: 'source')
            - 'target_col': column name for target (default: 'target')
            - 'weight_col': column name for weight (default: 'weight')
            - 'node_id_map': dict mapping indices to node ids

        Returns
        -------
        pd.DataFrame
            Edges DataFrame

        Examples
        --------
        >>> edgelist = np.array([[0, 1], [1, 2]])
        >>> df = edgelist_to_edges_dataframe(edgelist)
        >>> 'source' in df.columns and 'target' in df.columns
        True
        """
        ctx = ctx or {}
        source_col = ctx.get('source_col', 'source')
        target_col = ctx.get('target_col', 'target')
        weight_col = ctx.get('weight_col', 'weight')
        node_id_map = ctx.get('node_id_map', None)

        if len(edgelist) == 0:
            return pd.DataFrame(columns=[source_col, target_col])

        has_weights = edgelist.shape[1] >= 3

        # Build DataFrame
        data = {}

        if node_id_map:
            data[source_col] = [node_id_map.get(int(e[0]), int(e[0])) for e in edgelist]
            data[target_col] = [node_id_map.get(int(e[1]), int(e[1])) for e in edgelist]
        else:
            data[source_col] = edgelist[:, 0].astype(int)
            data[target_col] = edgelist[:, 1].astype(int)

        if has_weights:
            data[weight_col] = edgelist[:, 2]

        return pd.DataFrame(data)


# ============================================================================
# Edges DataFrame <-> nodes_and_links conversions
# ============================================================================

if _has_pandas:

    def edges_dataframe_to_nodes_and_links(df: pd.DataFrame, ctx: dict = None) -> dict:
        """
        Convert edges DataFrame to nodes_and_links dict.

        Parameters
        ----------
        df : pd.DataFrame
            Edges DataFrame
        ctx : dict, optional
            Context (same as edges_dataframe_to_edgelist)

        Returns
        -------
        dict
            Dict with 'nodes' and 'links' keys

        Examples
        --------
        >>> df = pd.DataFrame({'source': ['a', 'b'], 'target': ['b', 'c']})
        >>> result = edges_dataframe_to_nodes_and_links(df)
        >>> len(result['nodes']) == 3
        True
        """
        ctx = ctx or {}

        # Auto-detect columns
        source_col = ctx.get('source_col')
        if source_col is None:
            for col in ['source', 'src', 'from']:
                if col in df.columns:
                    source_col = col
                    break

        target_col = ctx.get('target_col')
        if target_col is None:
            for col in ['target', 'dst', 'to', 'dest']:
                if col in df.columns:
                    target_col = col
                    break

        if source_col is None or target_col is None:
            raise ValueError(f"Could not detect source/target columns")

        id_field = ctx.get('id_field', 'id')
        source_field = ctx.get('source_field', 'source')
        target_field = ctx.get('target_field', 'target')

        # Get unique nodes
        all_nodes = pd.concat([df[source_col], df[target_col]]).unique()
        nodes = [{id_field: str(node)} for node in all_nodes]

        # Build links
        links = []
        for _, row in df.iterrows():
            link = {
                source_field: str(row[source_col]),
                target_field: str(row[target_col]),
            }
            # Include other columns as attributes
            for col in df.columns:
                if col not in (source_col, target_col):
                    link[col] = row[col]
            links.append(link)

        return {'nodes': nodes, 'links': links}

    def nodes_and_links_to_edges_dataframe(
        nodes_and_links: dict, ctx: dict = None
    ) -> pd.DataFrame:
        """
        Convert nodes_and_links dict to edges DataFrame.

        Parameters
        ----------
        nodes_and_links : dict
            Dict with 'nodes' and 'links' keys
        ctx : dict, optional
            Context with optional keys:
            - 'source_col': column name for source (default: 'source')
            - 'target_col': column name for target (default: 'target')
            - 'source_field': field name in links for source (default: 'source')
            - 'target_field': field name in links for target (default: 'target')

        Returns
        -------
        pd.DataFrame
            Edges DataFrame

        Examples
        --------
        >>> data = {'nodes': [{'id': 'a'}, {'id': 'b'}], 'links': [{'source': 'a', 'target': 'b'}]}
        >>> df = nodes_and_links_to_edges_dataframe(data)
        >>> len(df)
        1
        """
        ctx = ctx or {}
        source_col = ctx.get('source_col', 'source')
        target_col = ctx.get('target_col', 'target')
        source_field = ctx.get('source_field', 'source')
        target_field = ctx.get('target_field', 'target')

        links = nodes_and_links.get('links', [])

        if not links:
            return pd.DataFrame(columns=[source_col, target_col])

        # Convert links to DataFrame rows
        rows = []
        for link in links:
            if isinstance(link, dict):
                row = dict(link)  # Copy
                # Rename source/target fields if needed
                if source_field != source_col and source_field in row:
                    row[source_col] = row.pop(source_field)
                if target_field != target_col and target_field in row:
                    row[target_col] = row.pop(target_field)
                rows.append(row)

        return pd.DataFrame(rows)


# ============================================================================
# Graph DataFrames <-> nodes_and_links conversions
# ============================================================================

if _has_pandas:

    def graph_dataframes_to_nodes_and_links(graph_dfs: dict, ctx: dict = None) -> dict:
        """
        Convert graph DataFrames (with edges and nodes) to nodes_and_links dict.

        Parameters
        ----------
        graph_dfs : dict
            Dict with 'edges' DataFrame and optionally 'nodes' DataFrame
        ctx : dict, optional
            Context (see edges_dataframe_to_nodes_and_links)

        Returns
        -------
        dict
            Dict with 'nodes' and 'links' keys

        Examples
        --------
        >>> edges_df = pd.DataFrame({'source': [0, 1], 'target': [1, 2]})
        >>> nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['A', 'B', 'C']})
        >>> graph_dfs = {'edges': edges_df, 'nodes': nodes_df}
        >>> result = graph_dataframes_to_nodes_and_links(graph_dfs)
        >>> len(result['nodes'])
        3
        """
        ctx = ctx or {}
        edges_df = graph_dfs['edges']
        nodes_df = graph_dfs.get('nodes', None)

        # Convert edges
        result = edges_dataframe_to_nodes_and_links(edges_df, ctx)

        # If nodes DataFrame is provided, use it to enrich node data
        if nodes_df is not None:
            id_field = ctx.get('id_field', 'id')
            id_col = ctx.get('id_col', 'id')

            # If DataFrame has 'id' column, use it; otherwise use index
            if id_col in nodes_df.columns:
                node_records = nodes_df.to_dict('records')
            else:
                # Use index as id
                node_records = []
                for idx, row in nodes_df.iterrows():
                    record = {id_field: str(idx)}
                    record.update(row.to_dict())
                    node_records.append(record)

            result['nodes'] = node_records

        return result

    def nodes_and_links_to_graph_dataframes(
        nodes_and_links: dict, ctx: dict = None
    ) -> dict:
        """
        Convert nodes_and_links dict to graph DataFrames.

        Parameters
        ----------
        nodes_and_links : dict
            Dict with 'nodes' and 'links' keys
        ctx : dict, optional
            Context (see nodes_and_links_to_edges_dataframe)

        Returns
        -------
        dict
            Dict with 'edges' and 'nodes' DataFrames

        Examples
        --------
        >>> data = {'nodes': [{'id': 0}, {'id': 1}], 'links': [{'source': 0, 'target': 1}]}
        >>> graph_dfs = nodes_and_links_to_graph_dataframes(data)
        >>> 'edges' in graph_dfs and 'nodes' in graph_dfs
        True
        """
        ctx = ctx or {}

        # Convert edges
        edges_df = nodes_and_links_to_edges_dataframe(nodes_and_links, ctx)

        # Convert nodes
        nodes = nodes_and_links.get('nodes', [])
        if nodes:
            nodes_df = pd.DataFrame(nodes)
        else:
            nodes_df = pd.DataFrame()

        return {'edges': edges_df, 'nodes': nodes_df}


# ============================================================================
# Register transformations
# ============================================================================

if _has_pandas:
    with suppress(ImportError, ModuleNotFoundError):
        # Edges DataFrame conversions
        register_transformation('edges_dataframe', 'edgelist', cost=0.4)(
            edges_dataframe_to_edgelist
        )
        register_transformation('edgelist', 'edges_dataframe', cost=0.4)(
            edgelist_to_edges_dataframe
        )
        register_transformation('edges_dataframe', 'nodes_and_links', cost=0.4)(
            edges_dataframe_to_nodes_and_links
        )
        register_transformation('nodes_and_links', 'edges_dataframe', cost=0.4)(
            nodes_and_links_to_edges_dataframe
        )

        # Graph DataFrames (with nodes) conversions
        register_transformation('graph_dataframes', 'nodes_and_links', cost=0.3)(
            graph_dataframes_to_nodes_and_links
        )
        register_transformation('nodes_and_links', 'graph_dataframes', cost=0.3)(
            nodes_and_links_to_graph_dataframes
        )
