from typing import Dict, List

import numpy as np
import time
import math
import datetime

class Krinsky:

    def __init__(self):
        self.n = 6
        self.k = 6
        self.states = self.k * self.n
        self.debug = False
        self.p: Dict[str, List[float]] = {
            "1": [0.0, 0.1, 0.2, 0.2, 0.2, 0.3],
            "2": [0.2, 0.0, 0.1, 0.2, 0.2, 0.3],
            "3": [0.3, 0.1, 0.0, 0.1, 0.2, 0.3],
            "4": [0.3, 0.2, 0.1, 0.0, 0.1, 0.3],
            "5": [0.4, 0.2, 0.2, 0.1, 0.0, 0.1],
            "6": [0.4, 0.2, 0.2, 0.1, 0.1, 0.0]
        }
        self.maxDelay = 100000

    # random environment where the best waiting time is calculated
    def environment(self, state):
        #print("environment: ", state)
        action = int((state-1)/self.n) + 1
        h = np.random.normal(2, 0.5)
        i = np.random.choice(range(1, 7), 1, False, self.p[str(action)])
        delay = (0.8 * i) + 0.4 * math.ceil(i / 2) + h
        if delay > self.maxDelay:
            return 1
        else:
            self.maxDelay = delay
            return 0

    def reward(self, state):
        if state % self.n == 0:
            state = state - (self.n - 1)
        else:
            state = state - state % self.n + 1
        #print("reward: ", state)
        return state


    def penalize(self, state):
        if state % self.n == 0:
            state = (state + self.n) % self.states
            if state == 0:
                state = self.n
        else:
            state += 1
        #print("penalize: ", state)
        return state

    def krinsky_automaton(self, batch, cutoff):
        a = np.zeros(6, dtype=int)
        learned_action_idx = None
        counts_test = 0
        action = np.random.randint(1, 7)
        state = action * self.n
        self.maxDelay = 10000

        for i in range(1, batch):
            result = self.environment(state)
            if result == 0:
                state = self.reward(state)
            elif result == 1:
                state = self.penalize(state)
            action_idx = int((state - 1) / self.n)
            a[action_idx] += 1
            if i == batch - cutoff:
                max = 0
                idx = None
                for j, v in enumerate(a):
                    if v > max:
                        max = v
                        idx = j
                learned_action_idx = idx
            elif i > batch - cutoff and action_idx == learned_action_idx:
                counts_test += 1
        accuracy = counts_test * 100 / cutoff
        return accuracy, self.maxDelay, str(learned_action_idx + 1)

    def main(self):
        start = datetime.datetime.now()
        total_accuracy = 0
        accuracy_count = 0
        total_delay = 0
        delay_count = 0
        learned_floors = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
        for i in range(100):
            accuracy, delay, learned_floor = self.krinsky_automaton(10000, 1000)
            total_accuracy += accuracy
            total_delay += delay
            if accuracy > 0:
                accuracy_count += 1
            if delay > 0:
                delay_count += 1
            learned_floors[learned_floor] += 1
        average_accuracy = total_accuracy/accuracy_count
        average_delay = total_delay/delay_count
        end = datetime.datetime.now()
        print("Krinsky")
        print("="*10)
        print("Time taken: ", end-start)
        print("Average_accuracy: {} % Average delay: {} seconds".format(average_accuracy, average_delay))
        print("Learned floor counts: ")
        print("1:", learned_floors["1"])
        print("2:", learned_floors["2"])
        print("3:", learned_floors["3"])
        print("4:", learned_floors["4"])
        print("5:", learned_floors["5"])
        print("6:", learned_floors["6"])


if __name__ == "__main__":
    krinsky = Krinsky()
    krinsky.main()
