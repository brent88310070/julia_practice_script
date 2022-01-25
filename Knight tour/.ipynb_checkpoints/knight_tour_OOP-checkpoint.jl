mutable struct Board
    matrix::Matrix{Int}
end

function Board(BOARD_SIZE::Int)
    mat = fill(0, BOARD_SIZE, BOARD_SIZE)
    return Board(mat)
end

mutable struct Knight
    location::Vector{Int64}
    step::Int64
end



Possible_steps(x) = [[x[1]-2, x[2]+1],
                     [x[1]-1, x[2]+2],
                     [x[1]+1, x[2]+2],
                     [x[1]+2, x[2]+1],
                     [x[1]+2, x[2]-1],
                     [x[1]+1, x[2]-2],
                     [x[1]-1, x[2]-2],
                     [x[1]-2, x[2]-1]]

Boundary_check(x, BOARD_SIZE) = 0 < x[1] < BOARD_SIZE+1 && 0 < x[2] < BOARD_SIZE+1

function Find_least_step(b::Board, random_walk_list)
    steps = 8
    best_loc = [0, 0]
    if isempty(random_walk_list)
        return []
    else
        for i in random_walk_list
            step_loc = Possible_steps(i)
            step_loc = [i for i in step_loc if Boundary_check(i, BOARD_SIZE)]
            step_loc = [i for i in step_loc if b.matrix[i[1], i[2]] < 1]
            if length(step_loc) <= steps
                steps = length(step_loc)
                best_loc = i
            end
        end
    end
    return best_loc
end

function Random_choose(k::Knight, b::Board)
    random_walk_list = Possible_steps(k.location)
    random_walk_list = [i for i in random_walk_list if Boundary_check(i, BOARD_SIZE)]
    random_walk_list = [i for i in random_walk_list if b.matrix[i[1], i[2]] < 1]
    location = Find_least_step(b, random_walk_list)
    return location
end

function Run_script()
    board = Board(BOARD_SIZE)
    start_location = rand(1:BOARD_SIZE, 2)
    step = 1
    man = Knight(start_location, step)
    board.matrix[man.location...] = man.step

    while (any(x->x==0, board.matrix))
        new_location = Random_choose(man, board)
        if isempty(new_location)
            board = Board(BOARD_SIZE)
            start_location = rand(1:BOARD_SIZE, 2)
            man.step = 1
            man.location = start_location
            board.matrix[man.location...] = man.step

        else
            man.step = man.step + 1
            board.matrix[new_location...] = man.step
            man.location = new_location
        end
    end
    return board.matrix
end



println("Input board size: ")
const BOARD_SIZE = parse(Int64, readline())

display(Run_script())


