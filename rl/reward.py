from __future__ import annotations

from typing import Dict, List, Tuple, Union

from core.game_state import GameState


WinnerType = Union[int, List[int], None]


def compute_score(state: GameState, nation_id: int) -> int:
    nation = state.nations[nation_id]
    return (
        nation.gold
        + nation.manpower
        + nation.production
        + len(nation.unlocked_techs) * 500
        + len(nation.unlocked_civics) * 500
    )


def determine_terminal_winner(state: GameState, max_turns_reached: bool = False) -> WinnerType:
    winner = state.check_winner()
    if winner is not None:
        return winner

    if not max_turns_reached:
        return None

    alive = [n.id for n in state.nations.values() if not n.is_defeated]
    if not alive:
        return None

    scores = {nation_id: compute_score(state, nation_id) for nation_id in alive}
    best = max(scores.values())
    tied = [nation_id for nation_id, score in scores.items() if score == best]
    if len(tied) == 1:
        return tied[0]
    return tied


def compute_reward(
    prev_score: int,
    next_score: int,
    learner_id: int,
    learner_name: str,
    next_state: GameState,
    winner: WinnerType,
    done: bool,
    resolved_logs: List[str],
    dense_reward_scale: float = 0.01,
    terminal_win_reward: float = 100.0,
    terminal_peace_reward: float = 50.0,
    terminal_loss_penalty: float = -100.0,
    annex_bonus: float = 20.0,
) -> Tuple[float, Dict[str, float]]:
    dense_delta = next_score - prev_score
    dense_reward = dense_reward_scale * dense_delta

    annex_count = sum(
        1
        for line in resolved_logs
        if "HAS FALLEN!" in line and f"{learner_name} annexed" in line
    )
    annex_reward = annex_bonus * annex_count

    terminal_reward = 0.0
    learner_defeated = next_state.nations[learner_id].is_defeated
    learner_won = False
    peace_victory = False

    if done:
        if isinstance(winner, list):
            learner_won = learner_id in winner
            peace_victory = learner_won
        else:
            learner_won = winner == learner_id

        if learner_defeated:
            terminal_reward = terminal_loss_penalty
        elif learner_won and peace_victory:
            terminal_reward = terminal_peace_reward
        elif learner_won:
            terminal_reward = terminal_win_reward
        else:
            terminal_reward = terminal_loss_penalty

    total_reward = dense_reward + annex_reward + terminal_reward
    components = {
        "dense_delta": float(dense_delta),
        "dense_reward": float(dense_reward),
        "annex_count": float(annex_count),
        "annex_reward": float(annex_reward),
        "terminal_reward": float(terminal_reward),
        "total_reward": float(total_reward),
    }
    return total_reward, components
