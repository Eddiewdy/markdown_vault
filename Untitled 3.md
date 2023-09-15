Function DFS(grap, start):

​	visited = set()

​	DFS_recursive(graph, start, visited)

Function DFS_recursive(graph, current, visited):

​	visited.add(current)

​	// visite

​	for neigh in graph[current]:
​		if neigh not in visited:

​			DFS_recursive(graph, neigh, visited)	



Function BFS(graph, start):

​	visited = set()

​	queue = Queue()

​	visited.add(start)

​	queue.enqueue(start)

​	while not queue.is_empty():
​		current = queue.dequeue()

​		for neigh in graph[current]:

​			if neigh not in visited:

​				visited.add(neigh)

​				queue.enqueue(neigh)



Function detect_loop(graph, current):

​	visited = set()

​	if curr is visited:

​		// loop detected

​	visited.add(current)

​	for neigh in graph[current]:

​		detect_loop(graph, neigh)

​	visited.remove(current)



Function detect_loop(graph, node, visited, in_recursion):

​	visited.add(node)

​	in_recursion.add(node)

​	for neigh in graph[node]:

​		if neigh not in visited:

​			DFS(graph, neigh, visited, in_recursion)

​		else if neigh in in_recursion:

​			return true

​	in_recursion.remove(node)

​	