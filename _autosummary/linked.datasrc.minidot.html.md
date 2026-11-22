# linked.datasrc.minidot

mini-dot: A very simple mini-language resembling graphviz dot

### Functions

| `iterize`(func)                                                                               |                                                |
|-----------------------------------------------------------------------------------------------|------------------------------------------------|
| [`mini_dot_to_graph_jdict`](#linked.datasrc.minidot.mini_dot_to_graph_jdict)(mini_dot, \*[, ...]) | Make a graph json dict from a mini-dot string. |
| `source_target_pairs_to_graph_jdict`(...[, ...])                                              |                                                |

### linked.datasrc.minidot.extract_nodes(string, pos=0, endpos=9223372036854775807)

Return a list of all non-overlapping matches of pattern in string.

### linked.datasrc.minidot.mini_dot_to_graph_jdict(mini_dot, , field_names={'id': 'id', 'links': 'links', 'nodes': 'nodes', 'source': 'source', 'target': 'target'})

Make a graph json dict from a mini-dot string.

```pycon
>>> mini_dot_to_graph_jdict('''
... 1 -> 2
... 2, 3 -> 5, 6, 7
... ''')
{'nodes': [{'id': '1'},
{'id': '2'},
{'id': '5'},
{'id': '6'},
{'id': '7'},
{'id': '3'}],
'links': [{'source': '1', 'target': '2'},
{'source': '2', 'target': '5'},
{'source': '2', 'target': '6'},
{'source': '2', 'target': '7'},
{'source': '3', 'target': '5'},
{'source': '3', 'target': '6'},
{'source': '3', 'target': '7'}]}
```
