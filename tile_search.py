from collections import deque

# 123
# 456
# 78X 

# 123
# 4X5
# 678

target_state = ["1","2","3","4","X","5","6","7","8"]



def serialize_state(a: list[str]) -> str:
    return ".".join(a)


def compare_states(a: list[str],b: list[str]) -> bool:
    return serialize_state(a) == serialize_state(b)


def create_new_state(state: list[str],position: int,new_position: int) -> list[str]:
    arr = state.copy()
    temp = arr[new_position]
    arr[new_position] = arr[position]
    arr[position] = temp
    return arr

def search(start: list[str]):
    states_to_search:deque[list[str]] = deque()
    states_to_search.append(start)
    searched_states = set([serialize_state(start)])
    
    iterations = 0
    while len(states_to_search) > 0:
        iterations += 1

        current_state = states_to_search.popleft()
        
        if compare_states(current_state,target_state):
            print(f"Solved after {iterations} iterations")
            print(current_state)
            return True
        
        position_in_state = current_state.index('X')

        possible_destinations = []
        if position_in_state == 0:
            possible_destinations.extend([1,3])
        elif position_in_state == 1:
            possible_destinations.extend([0,4,2])
        elif position_in_state == 2:
            possible_destinations.extend([2,5])
        elif position_in_state == 3:
            possible_destinations.extend([0,4,6])
        elif position_in_state == 4:
            possible_destinations.extend([1,5,7,3])
        elif position_in_state == 5:
            possible_destinations.extend([2,4,8])
        elif position_in_state == 6:
            possible_destinations.extend([3,7])
        elif position_in_state == 7:
            possible_destinations.extend([4,8,6])
        elif position_in_state == 8:
            possible_destinations.extend([5,7])

        for dest in possible_destinations:
            resulting_state = create_new_state(current_state,position_in_state,dest)

            if serialize_state(resulting_state) not in searched_states:
                states_to_search.append(resulting_state)
                searched_states.add(serialize_state(resulting_state))
        
    print(f"Failed to find solution after {iterations} iterations")

puzzle = ["1","X","3","4","5","6","7","8","2"]
search(puzzle)