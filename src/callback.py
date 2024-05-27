import regelum
from regelum.callback import Callback, HistoricalCallback, disable_in_jupyter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regelum.simulator

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
                        zip(self.observation_components_naming, output["observation"][0])
                    ),
                    **dict(
                        zip(self.state_components_naming, output["estimated_state"][0])
                    ),
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
                        (self.observation_components_naming, "observation"),
                        (self.state_components_naming, "estimated_state"),
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
    
    

class SimulatorStepLogger(Callback):
    """Records the state of the simulated system into `self.system_state`.

    Useful for animations that visualize motion of dynamical systems.
    """

    def __init__(self):
        super().__init__()
        self.times = []
        self.system_states = []
        self.observations = []

        self.reset()

    def reset(self):
        self.first_step = True
        self.data = None

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.simulator.Simulator)
            and method == "get_sim_step_data"
        )

    def is_done_collecting(self):
        return hasattr(self, "system_state")

    def on_function_call(self, obj, method, output):
        # self.system_state = obj.state
        # TIME
        self.times.append(output[0])

        # STATE
        system_state = output[1]
        system_state = system_state.reshape(system_state.size)
        # self.state_naming = obj.simulator.system.state_naming
        self.state_naming = obj.system.state_naming
        self.system_states.append(system_state)
        
        # OBSERVATION
        # observation = output[2] # NOISE OBSERVATION
        # True observation
        observation = obj.system.get_clean_observation(system_state)
        observation = observation.reshape(observation.size)
        self.observation_naming = obj.system.observation_naming
        self.observations.append(observation)

        # create dataframe 
        if self.first_step == True:
            column_names = ["time"]
            column_names.extend(self.observation_naming)
            column_names.extend(self.state_naming)
            self.data = pd.DataFrame(columns=column_names)
            self.first_step = False

        output_to_storage = [output[0]]
        output_to_storage.extend([observation[i] for i, _ in enumerate(observation)])
        output_to_storage.extend([system_state[i] for i, _ in enumerate(system_state)])
        self.data.loc[len(self.data.index)] = output_to_storage