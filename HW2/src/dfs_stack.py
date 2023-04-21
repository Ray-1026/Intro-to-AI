import csv

edgeFile = "edges.csv"


def dfs(start, end):
    # Begin your code (Part 2)
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
    Implement DFS using stack.
    1. Initialize an empty dictionary called 'parent', a list with start ID called 'stack', a
       set to mark the visited node ID called 'visited', and 'num_visited' and 'isfind' to 0.
    2. Pop the element from the back of the list 'stack', add it to the set 'visited', and add
       1 to the 'num_visited'.
    3. Explore the neighbors of the popped node. For each unvisited neighbors, add them to the
       'visited' and to the 'stack', and record the popped node and the distance in the dictionary
       'parent'.
    4. Repeat steps 2-3 until the popped node is the end we expected, and then set 'isfind' to 1.
    """
    parent = {}
    stack = [start]
    visited = set()
    visited.add(start)
    num_visited, isfind = 0, 0
    while stack:
        node = stack.pop()
        if node == end:
            isfind = 1
            break
        num_visited += 1
        for i, j in edge[node]:
            if i not in visited:
                stack.append(i)
                visited.add(i)
                parent[i] = (node, j)
    """
    Find the path and calculate the distance.
    1. Initialize an empty list 'path' and 'dist' to 0.
    2. If the variable 'isfind' is 1, which means DFS finds a path from 'start' to 'end' successfully,
       use the dictionary 'parent' to find the parent of the node and calculate the diatance. The
       value from the 'parent' is a tuple storing distance and parent ID, so add them to the 'dist'
       and 'path', respectively.
    3. Repeat step 2 until we find the start ID.
    4. Return 'path', 'dist', and 'num_visited'.
    """
    path = []
    dist = 0
    if isfind:
        path.append(end)
        while path[-1] != start:
            dist += parent[path[-1]][1]
            path.append(parent[path[-1]][0])
        path.reverse()
    return path, dist, num_visited
    # End your code (Part 2)


if __name__ == "__main__":
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f"The number of path nodes: {len(path)}")
    print(f"Total distance of path: {dist}")
    print(f"The number of visited nodes: {num_visited}")
