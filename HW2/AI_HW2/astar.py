import csv
import heapq

edgeFile = "edges.csv"
heuristicFile = "heuristic.csv"


def astar(start, end):
    # Begin your code (Part 4)
    """
    Read and process the data from the csv files.
    1. List the whole data from 'edges.csv'. Pop the first one, since it is the column name.
    2. In this case, I only need the first three of each row. Therefore, the first three
       data in each row are put in a dictionary called 'edge' to represent the adjacent list
       of this graph. Specifically, in my implementation, every node is a key with value being
       an empty list in dictionary. If there is a path in the graph, then add the tuple storing
       end ID and destination to the list where the start ID is its key.
    3. List the whole data from 'heuristic.csv', and find which elements in the rows are wanted.
    4. Put all the wanted values in each row in the dictionary called 'heuristic'.
    """
    edge = {}
    heuristic = {}
    with open(edgeFile) as file1, open(heuristicFile) as file2:
        csv1, csv2 = list(csv.reader(file1)), list(csv.reader(file2))
        csv1.pop(0)
        for i in csv1:
            num1, num2, distance = int(i[0]), int(i[1]), float(i[2])
            if num1 not in edge:
                edge[num1] = [(num2, distance)]
            else:
                edge[num1].append((num2, distance))
            if num2 not in edge:
                edge[num2] = []
        idx = next((i for i in range(1, 4) if csv2[0][i] == str(end)))
        csv2.pop(0)
        for i in csv2:
            heuristic[int(i[0])] = float(i[idx])
    """
    Implement A* search.
    1. Initialize an empty dictionary called 'parent', a heapified list like priority queue called
       'heap', an empty set called 'visited', and 'num_visited', 'isfind' and 'dist' to 0.
    2. Pop out the first tuple in the 'heap'. The tuple contains three elements, which represents
       the sum of distance from start node and the cost from heuristic function, the current node ID,
       and its parent.
    3. If the current node is unvisited, add the node to the 'visited' and add 1 to the 'num_visited'.
       According to its parent in the tuple, use 'parent' to record.
    4. Then, explore its neighbor nodes, and add the tuple storing the sum of distance from start to the
       neighbor and the cost from heuristic function, ID of the neighbor, and current node ID to the 'heap'.
    5. Repeat steps 2-5 until the end node is found, and then set 'isfind' to 1 and 'dist' to the first
       element of the tuple.
    """
    parent = {}
    heap = [(heuristic[start], start, None)]
    heapq.heapify(heap)
    visited = set()
    num_visited, isfind, dist = 0, 0, 0
    while heap:
        (cost, node, p) = heapq.heappop(heap)
        cost -= heuristic[node]
        if node not in visited:
            num_visited += 1
            visited.add(node)
            parent[node] = p
            if node == end:
                isfind, dist = 1, cost
                break
            for i, j in edge[node]:
                heapq.heappush(heap, (cost + j + heuristic[i], i, node))
    """
    Find the path.
    1. Initialize an empty list 'path'.
    2. If the variable 'isfind' is 1, which means A* search finds a path from 'start' to 'end'
       successfully, use the dictionary 'parent' to find the parent of the node. The value from
       the 'parent' is parent ID, so add it to the 'path'.
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
    # End your code (Part 4)


if __name__ == "__main__":
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f"The number of path nodes: {len(path)}")
    print(f"Total distance of path: {dist}")
    print(f"The number of visited nodes: {num_visited}")
