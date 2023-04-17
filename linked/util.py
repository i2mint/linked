"""Utils"""


def ordered_unique(iterable):
    seen = set()
    seen_add = seen.add
    return (x for x in iterable if not (x in seen or seen_add(x)))


def _nodes_from_links(links):
    def _yield_nodes_from_links():
        for link in links:
            yield link['source']
            yield link['target']

    return [{'id': x} for x in ordered_unique(_yield_nodes_from_links(links))]