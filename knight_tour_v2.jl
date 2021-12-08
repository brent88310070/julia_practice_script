const BOARD_SIZE = 8
const RANGE = 1:8

rand_pos() = rand(RANGE, 2)
boundary_check(x) = 0 < x[1] < BOARD_SIZE+1 && 0 < x[2] < BOARD_SIZE+1
possible_steps(x) = [[x[1]-2, x[2]+1],
                     [x[1]-1, x[2]+2],
                     [x[1]+1, x[2]+2],
                     [x[1]+2, x[2]+1],
                     [x[1]+2, x[2]-1],
                     [x[1]+1, x[2]-2],
                     [x[1]-1, x[2]-2],
                     [x[1]-2, x[2]-1]]

function find_least_step(random_walk_list, matrix)
    steps = 8
    best_loc = [0, 0]
    if isempty(random_walk_list)
        return []
    else
        for i in random_walk_list
            step_loc = possible_steps(i)
            step_loc = [i for i in step_loc if boundary_check(i)]
            step_loc = [i for i in step_loc if matrix[i[1], i[2]] < 1]
            if length(step_loc) <= steps
                steps = length(step_loc)
                best_loc = i
            end
        end
    end
    return best_loc
end

function random_choose(now_location, matrix)
    random_walk_list = possible_steps(now_location)
    random_walk_list = [i for i in random_walk_list if boundary_check(i)]
    random_walk_list = [i for i in random_walk_list if matrix[i[1], i[2]] < 1]
    location = find_least_step(random_walk_list, matrix)
    return location
end

function run_script()
    matrix = fill(0, BOARD_SIZE, BOARD_SIZE)
    start_point = rand_pos()
    step = 1
    matrix[start_point...] = step
    now_location = start_point

    while (any(x->x==0, matrix))
        new_location = random_choose(now_location, matrix)
        if isempty(new_location)
            matrix = fill(0, BOARD_SIZE, BOARD_SIZE)
            start_point = rand_pos()
            step = 1
            matrix[start_point...] = step
            now_location = start_point
        else
            step = step + 1
            matrix[new_location...] = step
            now_location = new_location
        end
    end
    return matrix
end

display(run_script())


