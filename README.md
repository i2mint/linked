# linked

Create and transform graphs

To install:	```pip install linked```

# Examples

## mini_dot_to_graph_jdict

```python
>>> from linked import mini_dot_to_graph_jdict
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