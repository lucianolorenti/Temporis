from copy import copy 

def root_nodes(final_step):
    visited = set([final_step])
    to_process = copy(final_step.previous)

    while len(to_process) > 0:
        t = to_process.pop()
        if t not in visited:
            visited.add(t)
            to_process.extend(t.previous)

    return [n for n in visited if len(n.previous) == 0]

def dfs_iterator(final_step):
    visited = set([])    
    Q = copy(root_nodes(final_step))
    while len(Q) > 0:
        node = Q.pop()
        if node in visited:
            continue  
        visited.add(node)
        yield node        
      
        if node.next is None:
            continue
        Q.append(node.next)


def topological_sort_iterator(final_step):
    in_degree = {}
    for node in dfs_iterator(final_step):
        if node not in in_degree:
            in_degree[node] = 0
        if node.next is None:
            continue
        if node.next not in in_degree:
            in_degree[node.next] = 0
        in_degree[node.next] += 1 
    Q = root_nodes(final_step)    
    while len(Q) > 0:    
    
        node = Q.pop(0)
        yield node
        if node.next is None:
            continue
        in_degree[node.next] -= 1
        if in_degree[node.next] == 0:
            Q.append(node.next)
