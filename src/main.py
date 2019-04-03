from src.policy import AI
from src.game import BatchGame
from src.play_state import PlayState


def play(ai: AI):
    game = BatchGame(batch_size=1)
    while game.cur_state.winners()[0] is None:
        if ai.player == game.cur_state.turn:
            moves = ai.next_moves(game.cur_state)
            game = game.add_state(game.cur_state.play_at(moves))
        else:
            print(game.cur_state)
            next_game = None
            while next_game is None:
                try:
                    j = int(input("Which column would you like to play?")) - 1
                    next_game = game.add_state(game.cur_state.play_at([j]))
                except:
                    pass
            game = next_game
    print("The winner is: %s" % game.cur_state.winners()[0])
    print(game.cur_state)
    return game


def train(ai1: AI, ai2: AI):
    assert ai1.player != ai2.player, "AIs must play different colors"
    total_num_games = 1
    game_length = 100
    for i in range(total_num_games):
        games = BatchGame(batch_size=128)
        print("\ntraining on game %d of %d" % (i, total_num_games))
        for j in range(game_length):
            n = int(10 * j/game_length)
            print("\r"+"*"*n, end="")
            winners = games.cur_state.winners()
            reset_games = [w is not None for w in winners]
            if ai1.player == games.cur_state.turn:
                moves = ai1.next_moves(games.cur_state)
                games = games.add_state(games.cur_state.play_at(moves, reset_games))
            else:
                moves = ai2.next_moves(games.cur_state)
                games = games.add_state(games.cur_state.play_at(moves, reset_games))
        ai1.learn_from_games(games, verbose=True)
        ai2.learn_from_games(games, verbose=True)


if __name__ == "__main__":
    aio = AI.load("Checkpoints-20190402-152625/O-player", PlayState.O)
    container = aio._pi._container
    aix = AI.load("Checkpoints-20190402-152625/X-player", PlayState.X)
    train(aio, aix)
    games = play(aix)
    aio.learn_from_games(games)
    aio.save("O-player")
    aix.save("X-player")