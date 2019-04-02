from src.batch_policy import AI
from src.batch_game import BatchGameState, BatchGame, Player
from src.game import Game


def play(ai: AI):
    game = Game()
    while game.cur_state.winner() is None:
        if ai.player == game.cur_state.turn:
            move = ai.next_move(game.cur_state)
            game = game.add_state(game.cur_state.play_at(move))
        else:
            print(game.cur_state)
            next_game = None
            while next_game is None:
                try:
                    j = int(input("Which column would you like to play?")) - 1
                    next_game = game.add_state(game.cur_state.play_at(j))
                except:
                    pass
            game = next_game
    print("The winner is: %s" % game.cur_state.winner())
    print(game.cur_state)
    return game


def train(ai1: AI, ai2: AI):
    assert ai1.player != ai2.player, "AIs must play different colors"
    total_num_games = 100
    game_length = 100
    for i in range(total_num_games):
        game = BatchGame()
        print("\ntraining on game %d of %d" % (i, total_num_games))
        for j in range(game_length):
            n = int(10 * j/game_length)
            print("\r"+"*"*n, end="")
            winners = game.cur_state.winners()
            reset_games = [w is not None for w in winners]
            if ai1.player == game.cur_state.turn:
                moves = ai1.next_moves(game.cur_state)
                game = game.add_state(game.cur_state.play_at(moves, reset_games))
            else:
                moves = ai2.next_moves(game.cur_state)
                game = game.add_state(game.cur_state.play_at(moves, reset_games))
        ai1.learn_from_games(game, verbose=True)
        ai2.learn_from_games(game, verbose=True)


if __name__ == "__main__":
    aio = AI.load("Checkpoints-20190402-113002/O-player", Player.O)
    container = aio._pi._container
    aix = AI.load("Checkpoints-20190402-113002/X-player", Player.X)
    train(aio, aix)
    game = play(aio)
    aio.learn_from_games(game.map(BatchGameState.from_game_state))
    aio.save("O-player")
    aix.save("X-player")