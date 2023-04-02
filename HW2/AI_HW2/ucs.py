import csv
import heapq

edgeFile = "edges.csv"


def ucs(start, end):
    # Begin your code (Part 3)
    """
    Read and process the data from the csv file.
    1. List the whole data from 'edges.csv'. Pop the first one, since it is the column name.
    2. In this case, I only need the first three of each row. Therefore, the first three
       data in each row are put in a dictionary called 'edge' to represent the adjacent list
       of this graph. Specifically, in my implementation, every node is a key with value being
       an empty list in dictionary. If there is a path in the graph, then add the tuple storing
       end ID and destination to the list where the start ID is its key.
    """
    edge = {}
    with open(edgeFile) as file:
        csvfile = list(csv.reader(file))
        csvfile.pop(0)
        for i in csvfile:
            num1, num2, distance = int(i[0]), int(i[1]), float(i[2])
            if num1 not in edge:
                edge[num1] = [(num2, distance)]
            else:
                edge[num1].append((num2, distance))
            if num2 not in edge:
                edge[num2] = []
    """
    Implement UCS.
    1. Initialize an empty dictionary called 'parent', a heapified list like priority queue called
       'heap', an empty set called 'visited', and 'num_visited', 'isfind' and 'dist' to 0.
    2. Pop out the first tuple in the 'heap'. The tuple contains three elements, which represents
       the distance from start node, the current node ID, and its parent.
    3. If the current node is unvisited, add the node to the 'visited' and add 1 to the 'num_visited'.
       According to its parent in the tuple, use 'parent' to record.
    4. Then, explore its neighbor nodes, and add the tuple storing the distance from start to the
       neighbor, ID of the neighbor, and current node ID to the 'heap'.
    5. Repeat steps 2-5 until the end node is found, and then set 'isfind' to 1 and 'dist' to the first
       element of the tuple.
    """
    parent = {}
    heap = [(0, start, None)]
    heapq.heapify(heap)
    visited = set()
    num_visited, isfind, dist = 0, 0, 0
    while heap:
        (cost, node, p) = heapq.heappop(heap)
        if node == end:
            isfind, dist = 1, cost
            parent[node] = p
            break
        if node not in visited:
            num_visited += 1
            visited.add(node)
            parent[node] = p
            for i, j in edge[node]:
                heapq.heappush(heap, (cost + j, i, node))
    """
    Find the path.
    1. Initialize an empty list 'path'.
    2. If the variable 'isfind' is 1, which means UCS finds a path from 'start' to 'end' successfully,
       use the dictionary 'parent' to find the parent of the node. The value from the 'parent' is parent
       ID, so add it to the 'path'.
    3. Repeat step 2 until we find the start ID.
    4. Return 'path', 'dist', and 'num_visited'.
    """
    path = []
    if isfind:
        path.append(end)
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
    return path, dist, num_visited
    # End your code (Part 3)


if __name__ == "__main__":
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f"The number of path nodes: {len(path)}")
    print(f"Total distance of path: {dist}")
    print(f"The number of visited nodes: {num_visited}")
