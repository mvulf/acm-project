import regelum
from regelum.callback import HistoricalCallback, disable_in_jupyter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class HistoricalDataCallback(HistoricalCallback):
    """A callback which allows to store desired data collected among different runs inside multirun execution runtime."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of HistoricalDataCallback."""
        super().__init__(*args, **kwargs)
        self.cooldown = 0.0

        self.observation_components_naming = None
        self.action_components_naming = None
        self.state_components_naming = None

    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, regelum.scenario.Scenario) and (
            method == "post_compute_action" or method == "dump_data_buffer"
        )

    def on_function_call(self, obj, method, output):
        if self.observation_components_naming is None:
            self.observation_components_naming = (
                [
                    f"observation_{i + 1}"
                    for i in range(obj.simulator.system.dim_observation)
                ]
                if obj.simulator.system.observation_naming is None
                else obj.simulator.system.observation_naming
            )

        if self.action_components_naming is None:
            self.action_components_naming = (
                [f"action_{i + 1}" for i in range(obj.simulator.system.dim_inputs)]
                if obj.simulator.system.inputs_naming is None
                else obj.simulator.system.inputs_naming
            )

        if self.state_components_naming is None:
            self.state_components_naming = (
                [f"state_{i + 1}" for i in range(obj.simulator.system.dim_state)]
                if obj.simulator.system.state_naming is None
                else obj.simulator.system.state_naming
            )

        if method == "post_compute_action":
            self.add_datum(
                {
                    **{
                        "time": output["time"],
                        "running_objective": output["running_objective"],
                        "current_value": output["current_value"],
                        "episode_id": output["episode_id"],
                        "iteration_id": output["iteration_id"],
                    },
                    **dict(zip(self.action_components_naming, output["action"][0])),
                    **dict(
                        zip(self.observation_components_naming, output["estimated_state"][0])
                    ),
                    # **dict(
                    #     zip(self.state_components_naming, output["estimated_state"][0])
                    # ),
                }
            )
        elif method == "dump_data_buffer":
            _, data_buffer = output
            self.data = pd.concat(
                [
                    data_buffer.to_pandas(
                        keys={
                            "time": float,
                            "running_objective": float,
                            "current_value": float,
                            "episode_id": int,
                            "iteration_id": int,
                        }
                    )
                ]
                + [
                    pd.DataFrame(
                        columns=columns,
                        data=np.array(
                            data_buffer.to_pandas([key]).values.tolist(),
                            dtype=float,
                        ).squeeze(),
                    )
                    for columns, key in [
                        (self.action_components_naming, "action"),
                        (self.state_components_naming, "estimated_state"),
                        # (self.state_components_naming, "estimated_state"),
                    ]
                ],
                axis=1,
            )

    @disable_in_jupyter
    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        if episodes_total == 1:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}"
        else:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}_ep_{str(episode_number).zfill(5)}"
        self.save_plot(identifier)
        self.dump_and_clear_data(identifier)

    def plot(self, name=None):
        if regelum.main.is_clear_matplotlib_cache_in_callbacks:
            plt.clf()
            plt.cla()
            plt.close()

        if not name:
            name = self.__class__.__name__

        axes = (
            self.data[
                self.observation_components_naming + self.action_components_naming + ["time"]
            ]
            .set_index("time")
            .plot(subplots=True, grid=True, xlabel="time", title=name, legend=False)
        )
        for ax, label in zip(
            axes, self.observation_components_naming + self.action_components_naming
        ):
            ax.set_ylabel(label)

        return axes[0].figure