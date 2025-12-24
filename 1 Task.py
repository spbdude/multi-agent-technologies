import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Deque, Any


COST_SEND_MSG = 0.1
COST_SEND_TO_CENTER = 1000.0
COST_ARITH = 0.01
COST_MEM_CELL = 0.1


@dataclass
class Message:
    sender: str
    value: float
    iteration: int


class CostMeter:
    def __init__(self):
        self.msg_sent = 0
        self.msg_to_center = 0
        self.arith_ops = 0
        self.mem_cells = 0

    def add_send(self, n=1):
        self.msg_sent += n

    def add_send_to_center(self, n=1):
        self.msg_to_center += n

    def add_arith(self, n=1):
        self.arith_ops += n

    def add_mem_cells(self, n):
        self.mem_cells += n

    def total_cost(self) -> float:
        return (self.msg_sent * COST_SEND_MSG
                + self.msg_to_center * COST_SEND_TO_CENTER
                + self.arith_ops * COST_ARITH
                + self.mem_cells * COST_MEM_CELL)

    def report(self, iterations: int) -> str:
        return "\n".join([
            "===== COST REPORT =====",
            f"iterations: {iterations}",
            f"messages sent (peer-to-peer): {self.msg_sent}  -> cost {self.msg_sent * COST_SEND_MSG:.4f}",
            f"messages sent to center:      {self.msg_to_center}  -> cost {self.msg_to_center * COST_SEND_TO_CENTER:.4f}",
            f"arith ops (+,/,...):          {self.arith_ops}  -> cost {self.arith_ops * COST_ARITH:.4f}",
            f"memory cells (numbers):       {self.mem_cells}  -> cost {self.mem_cells * COST_MEM_CELL:.4f}",
            f"TOTAL COST: {self.total_cost():.4f}",
            "======================="
        ])


class Agent:
    def __init__(self, name: str, initial_value: float):
        self.name = name
        self.value = float(initial_value)
        self.neighbors: List[str] = []
        self.inbox: Deque[Message] = deque()

    def send_value(self, iteration: int, network: Dict[str, "Agent"], cost: CostMeter):
        for nb in self.neighbors:
            network[nb].inbox.append(Message(sender=self.name, value=self.value, iteration=iteration))
            cost.add_send(1)

    def compute_next_value(self, iteration: int, cost: CostMeter) -> float:
        needed = len(self.neighbors)
        got_values = []
        while self.inbox and self.inbox[0].iteration == iteration:
            msg = self.inbox.popleft()
            got_values.append(msg.value)

        if len(got_values) != needed:
            raise RuntimeError(f"{self.name}: expected {needed} msgs at iter={iteration}, got {len(got_values)}")

        deg = needed
        s = 0.0
        for v in got_values:
            s += v
        if deg > 0:
            cost.add_arith(deg)
        s2 = s + self.value
        cost.add_arith(1)
        denom = deg + 1
        cost.add_arith(1)
        return s2 / denom


def build_connected_random_graph(n: int, min_deg=2, max_deg=3, seed=None) -> Dict[int, List[int]]:
    rnd = random.Random(seed)
    adj = {i: set() for i in range(n)}
    nodes = list(range(n))
    rnd.shuffle(nodes)
    for idx in range(1, n):
        a = nodes[idx]
        b = nodes[rnd.randrange(0, idx)]
        adj[a].add(b)
        adj[b].add(a)

    def can_add_edge(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        if len(adj[u]) >= max_deg or len(adj[v]) >= max_deg:
            return False
        return True

    attempts = 0
    max_attempts = n * n * 20
    while attempts < max_attempts:
        attempts += 1
        low = [i for i in range(n) if len(adj[i]) < min_deg]
        if not low:
            break
        u = rnd.choice(low)
        v = rnd.randrange(n)
        if can_add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)

    return {i: sorted(list(neis)) for i, neis in adj.items()}


def consensus(
    n_agents: int = None,
    seed: int = None,
    eps: float = 1e-6,
    max_iters: int = 10000,
    min_deg: int = 2,
    max_deg: int = 3
):
    rnd = random.Random(seed)

    if n_agents is None:
        n_agents = rnd.randint(5, 15)

    initial = [rnd.randint(0, 100) for _ in range(n_agents)]
    true_mean = sum(initial) / n_agents

    graph = build_connected_random_graph(n_agents, min_deg=min_deg, max_deg=max_deg, seed=rnd.randrange(10**9))

    agents: Dict[str, Agent] = {}
    for i in range(n_agents):
        agents[f"agent{i}"] = Agent(f"agent{i}", initial[i])

    for i in range(n_agents):
        agents[f"agent{i}"].neighbors = [f"agent{j}" for j in graph[i]]

    cost = CostMeter()
    cost.add_mem_cells(n_agents)

    it = 0
    while it < max_iters:
        for a in agents.values():
            a.send_value(it, agents, cost)

        new_values: Dict[str, float] = {}
        max_delta = 0.0
        for name, a in agents.items():
            nv = a.compute_next_value(it, cost)
            new_values[name] = nv
            d = abs(nv - a.value)
            if d > max_delta:
                max_delta = d

        for name, nv in new_values.items():
            agents[name].value = nv

        it += 1
        if max_delta < eps:
            break

    cost.add_send_to_center(1)

    final_values = [a.value for a in agents.values()]
    consensus_value = sum(final_values) / len(final_values)
    abs_err = abs(consensus_value - true_mean)

    print("===== CONFIG =====")
    print(f"agents: {n_agents}")
    print(f"eps: {eps}")
    print(f"max_iters: {max_iters}")
    print(f"degree target: {min_deg}-{max_deg}")
    print("\n===== INITIAL VALUES =====")
    for i, v in enumerate(initial):
        print(f"agent{i}: {v}")
    print("\n===== RANDOM CONNECTIVITY (neighbors) =====")
    for i in range(n_agents):
        print(f"agent{i}: {graph[i]}")

    print()
    print(cost.report(iterations=it))
    print("\n===== RESULT =====")
    print(f"consensus mean: {consensus_value:.10f}")
    print(f"true mean:      {true_mean:.10f}")
    print(f"abs error:      {abs_err:.10f}")


if __name__ == "__main__":
    consensus(seed=58)