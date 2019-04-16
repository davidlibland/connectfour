from src.ai import AI
from src.game import BatchGame
from src.play_state import PlayState


def play(ai: AI, turn=PlayState.X):
    game = BatchGame(batch_size=1, turn=turn)
    while game.cur_state.winners()[0] is None:
        if ai.player == game.cur_state.turn:
            # Move and learn
            moves = ai.next_moves(game.cur_state)
            if game.liat and game.liat.liat:
                ai.learn_from_update(
                    game.liat.liat.cur_state,
                    game.liat.prev_action,
                    game.cur_state
                )
            game = game.play_at(moves)
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
    if game.liat and game.liat.liat:
        ai.learn_from_update(
            game.liat.liat.cur_state,
            game.liat.prev_action,
            game.cur_state
        )
    return game


def train(ai1: AI, ai2: AI, save_interval=10):
    assert ai1.player != ai2.player, "AIs must play different colors"
    total_num_games = 2
    game_length = 100
    for i in range(total_num_games):
        games = BatchGame(batch_size=16)
        print("\ntraining on game %d of %d" % (i+1, total_num_games))
        for j in range(game_length):
            n = int(10 * j/game_length)
            print("\r"+"*"*n, end="")
            winners = games.cur_state.winners()
            reset_games = [w is not None for w in winners]

            # Determine the AI
            if ai1.player == games.cur_state.turn:
                ai = ai1
            else:
                ai = ai2

            # Move and learn
            moves = ai.next_moves(games.cur_state)
            if games.liat and games.liat.liat:
                ai.learn_from_update(
                    games.liat.liat.cur_state,
                    games.liat.prev_action,
                    games.cur_state
                )
            games = games.play_at(moves, reset_games)
        if i != 0 and i % save_interval == 0:
            ai1.save()
            ai2.save()


if __name__ == "__main__":
    aio = AI.load("torch_ckpt/o_player_[0-9]*.pt", PlayState.O)
    aix = AI.load("torch_ckpt/x_player_[0-9]*.pt", PlayState.X)
    train(aio, aix)
    play_again = "y"
    while play_again == "y":
        player_txt = ""
        while player_txt not in ["x", "o"]:
            player_txt = input("Play as X or O? ").lower()
        if player_txt == "x":
            ai = aio
        else:
            ai = aix
        games = play(ai)
        # ai.learn_from_games(games)
        play_again = input("Would you like to play again? ").lower()
    aio.save()
    aix.save()
