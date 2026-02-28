#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Round Finished Stage."""

import math
import time

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class RoundFinishedStage(Stage):
    """Round Finished Stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "RoundFinishedStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        learner: Learner | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        aggregator: Aggregator | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on RoundFinishedStage.")

        # Set Next Round
        aggregator.clear()
        state.increase_round()

        # Round barrier: wait for >= 70% of neighbors to have finished the previous round
        # before advancing, so that fast nodes do not outpace slow ones.
        neighbors = list(communication_protocol.get_neighbors(only_direct=False))
        if neighbors:
            quorum = max(1, math.ceil(len(neighbors) * 0.9))
            deadline = time.time() + Settings.training.VOTE_TIMEOUT
            ready = 0
            while time.time() < deadline:
                ready = sum(1 for n in neighbors if state.nei_status.get(n, -1) >= state.round - 1)
                if ready >= quorum:
                    break
                time.sleep(1)
            else:
                logger.warning(state.addr, f"Round barrier timeout. {ready}/{len(neighbors)} neighbors ready.")

        # Next Step or Finish
        logger.info(
            state.addr,
            f"🎉 Round {state.round} of {state.total_rounds} finished.",
        )
        if state.round is None or state.total_rounds is None:
            raise ValueError("Round or total rounds not set.")

        if state.round < state.total_rounds:
            return StageFactory.get_stage("VoteTrainSetStage")
        else:
            # At end, all nodes compute metrics
            RoundFinishedStage.__evaluate(state, learner, communication_protocol)
            # Finish
            state.clear()
            logger.info(state.addr, "😋 Training finished!!")
            return None

    @staticmethod
    def __evaluate(state: NodeState, learner: Learner, communication_protocol: CommunicationProtocol) -> None:
        logger.info(state.addr, "🔬 Evaluating...")
        results = learner.evaluate()
        logger.info(state.addr, f"📈 Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )
