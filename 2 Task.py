import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Deque, Tuple, Optional


COST_SEND_MSG = 0.1
COST_SEND_TO_CENTER = 1000.0
COST_ARITH = 0.01
COST_MEM_CELL = 0.1


@dataclass
class Message:
    sender: str
    receiver: str
    value: float
    sent_iter: int
    deliver_iter: int


class CostMeter:
    def __init__(self):
        self.msg_sent = 0
        self.msg_lost = 0
        self.msg_delayed = 0
        self.msg_to_center = 0
        self.arith_ops = 0
        self.mem_cells = 0

    def add_send(self, n=1):
        self.msg_sent += n

    def add_lost(self, n=1):
        self.msg_lost += n

    def add_delayed(self, n=1):
        self.msg_delayed += n

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

    def report(self, iterations: int, offline_total_steps: int) -> str:
        return "\n".join([
            f"iterations: {iterations}",
            f"offline agent-steps: {offline_total_steps}",
            "",
            f"messages sent (attempted p2p): {self.msg_sent}  -> cost {self.msg_sent * COST_SEND_MSG:.4f}",
            f"messages lost (10% model):     {self.msg_lost}",
            f"messages delayed (1-2 iters):  {self.msg_delayed}",
            f"messages sent to center:       {self.msg_to_center}  -> cost {self.msg_to_center * COST_SEND_TO_CENTER:.4f}",
            f"arith ops (+,-,*,/,...):       {self.arith_ops}  -> cost {self.arith_ops * COST_ARITH:.4f}",
            f"memory cells (numbers):        {self.mem_cells}  -> cost {self.mem_cells * COST_MEM_CELL:.4f}",
            f"TOTAL COST: {self.total_cost():.4f}",
            "==================================================="
        ])


class Agent:
    def __init__(self, name: str, initial_value: float):
        self.name = name
        self.x = float(initial_value)
        self.neighbors: List[str] = []
        self.inbox: Deque[Message] = deque()
        self.offline_until_iter: int = -1

    def is_offline(self, t: int) -> bool:
        return t <= self.offline_until_iter

    def maybe_go_offline(self, t: int, rnd: random.Random, p_offline_start: float):
        if self.is_offline(t):
            return
        if rnd.random() < p_offline_start:
            duration = rnd.randint(1, 2)
            self.offline_until_iter = t + duration  # offline на t..t+duration включительно

    def receive_delivered(self, delivered: List[Message]):
        for m in delivered:
            self.inbox.append(m)

    def lvm_update(self, t: int, alpha: float, cost: CostMeter) -> float:
        if self.is_offline(t):
            self.inbox.clear()
            return self.x
        vals = []
        while self.inbox and self.inbox[0].deliver_iter <= t:
            msg = self.inbox.popleft()
            vals.append(msg.value)
        if not vals:
            return self.x
        s = 0.0
        for v in vals:
            s += (v - self.x)
        cost.add_arith(2 * len(vals))
        dx = alpha * s
        cost.add_arith(1)  # *
        new_x = self.x + dx
        cost.add_arith(1)  # +
        return new_x


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

    def can_add(u, v):
        if u == v or v in adj[u]:
            return False
        if len(adj[u]) >= max_deg or len(adj[v]) >= max_deg:
            return False
        return True

    attempts = 0
    max_attempts = n * n * 30
    while attempts < max_attempts:
        attempts += 1
        low = [i for i in range(n) if len(adj[i]) < min_deg]
        if not low:
            break
        u = rnd.choice(low)
        v = rnd.randrange(n)
        if can_add(u, v):
            adj[u].add(v)
            adj[v].add(u)

    return {i: sorted(list(neis)) for i, neis in adj.items()}


class ImpairedNetwork:
    def __init__(self, rnd: random.Random, p_loss: float = 0.1, delay_min: int = 1, delay_max: int = 2):
        self.rnd = rnd
        self.p_loss = p_loss
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.schedule: Dict[int, List[Message]] = defaultdict(list)

    def send(self, sender: str, receiver: str, value: float, t: int, cost: CostMeter):
        cost.add_send(1)

        # loss
        if self.rnd.random() < self.p_loss:
            cost.add_lost(1)
            return

        d = self.rnd.randint(self.delay_min, self.delay_max)
        if d > 0:
            cost.add_delayed(1)
        deliver_t = t + d
        self.schedule[deliver_t].append(Message(sender=sender, receiver=receiver, value=value, sent_iter=t, deliver_iter=deliver_t))

    def deliver(self, t: int) -> Dict[str, List[Message]]:
        delivered = self.schedule.pop(t, [])
        by_receiver: Dict[str, List[Message]] = defaultdict(list)
        for m in delivered:
            by_receiver[m.receiver].append(m)
        return by_receiver


def run_task2_lvm(
    seed: int = 42,
    n_agents: Optional[int] = None,
    eps: float = 1e-6,
    max_iters: int = 20000,
    min_deg: int = 2,
    max_deg: int = 3,
    p_offline_start: float = 0.03,
    p_loss: float = 0.10
):
    rnd = random.Random(seed)

    if n_agents is None:
        n_agents = rnd.randint(5, 15)

    initial = [rnd.randint(0, 100) for _ in range(n_agents)]
    true_mean = sum(initial) / n_agents

    graph = build_connected_random_graph(n_agents, min_deg=min_deg, max_deg=max_deg, seed=rnd.randrange(10**9))

    agents: Dict[str, Agent] = {f"agent{i}": Agent(f"agent{i}", initial[i]) for i in range(n_agents)}
    for i in range(n_agents):
        agents[f"agent{i}"].neighbors = [f"agent{j}" for j in graph[i]]

    alpha = 0.2
    cost = CostMeter()
    cost.add_mem_cells(n_agents)
    net = ImpairedNetwork(rnd=rnd, p_loss=p_loss, delay_min=1, delay_max=2)
    offline_total_steps = 0
    it = 0

    while it < max_iters:
        for a in agents.values():
            a.maybe_go_offline(it, rnd, p_offline_start)
        offline_total_steps += sum(1 for a in agents.values() if a.is_offline(it))
        for a in agents.values():
            if a.is_offline(it):
                continue
            for nb in a.neighbors:
                net.send(sender=a.name, receiver=nb, value=a.x, t=it, cost=cost)

        delivered_map = net.deliver(it)
        for recv_name, msgs in delivered_map.items():
            if recv_name in agents:
                agents[recv_name].receive_delivered(sorted(msgs, key=lambda m: m.deliver_iter))

        new_x: Dict[str, float] = {}
        max_delta = 0.0
        for name, a in agents.items():
            nx = a.lvm_update(it, alpha, cost)
            new_x[name] = nx
            d = abs(nx - a.x)
            if d > max_delta:
                max_delta = d

        for name, nx in new_x.items():
            agents[name].x = nx

        xs = [a.x for a in agents.values()]
        spread = max(xs) - min(xs)

        it += 1
        if max_delta < eps and spread < eps:
            break
    cost.add_send_to_center(1)

    xs = [a.x for a in agents.values()]
    consensus_est = sum(xs) / len(xs)
    abs_err = abs(consensus_est - true_mean)

    print("===== TASK 2 =====")
    print(f"seed: {seed}")
    print(f"agents: {n_agents}")
    print(f"alpha: {alpha}")
    print(f"eps: {eps}, max_iters: {max_iters}")
    print(f"loss prob: {p_loss}")
    print(f"offline start prob/iter: {p_offline_start}")
    print(f"delay: 1-2 iterations")
    print("\nInitial values:")
    for i, v in enumerate(initial):
        print(f"  agent{i}: {v}")

    print("\nNeighbors:")
    for i in range(n_agents):
        print(f"  agent{i}: {graph[i]}")

    print()
    print(cost.report(iterations=it, offline_total_steps=offline_total_steps))
    print("\n===== RESULT =====")
    print(f"consensus estimate: {consensus_est:.10f}")
    print(f"true mean:         {true_mean:.10f}")
    print(f"abs error:         {abs_err:.10f}")


if __name__ == "__main__":
    run_task2_lvm(seed=58)