from typing import List
import math
import numpy as np
from state import GraphState, ActorOutput, Action
from vima_interface import VIMAInterface, VIMAAction
from dataclasses import dataclass
from enum import Enum


class ActorMode(Enum):
    """Enumeration of actor operation modes"""
    ORACLE = "oracle"  # Use VIMA's oracle actions
    PIPELINE = "pipeline"  # Use pipeline-generated actions


@dataclass
class ActorConfig:
    """Configuration for actor node operation mode"""
    mode: ActorMode = ActorMode.PIPELINE  # Default to pipeline actions


def actor_node(state: GraphState, config: ActorConfig = None) -> dict:
    """
    Convert thinker pick-place tuples into VIMA actions and execute them using the shared VIMA interface.

    Args:
        state: Current graph state containing thinker output and VIMA interface
        config: Configuration specifying whether to use oracle or pipeline actions
               (defaults to ActorMode.PIPELINE)

    Returns:
        Updated state with actor output and fresh observation
    """
    # Use default config if none provided
    if config is None:
        config = ActorConfig(mode=ActorMode.PIPELINE)

    print(f"🔍 Actor node called with state type: {type(state)}, mode: {config.mode.value}")
    if state is None:
        print("❌ Actor node received None state!")
        return {'actor_output': {}}

    try:
        current_prompt = state['current_prompt']
        thinker_data : dict = state['thinker_output'].get(current_prompt, {})

        decision = thinker_data.get('decision', 'FAILURE')
        pick_place_tuples = thinker_data.get('chosen_grasp_points', [])
        reasoning = thinker_data.get('reasoning', 'No reasoning provided')
        rotation_degrees_list = thinker_data.get('rotation_degrees', [])

        # Get the shared VIMA interface from state (created in graph_setup)
        vima_interface = state.get('vima_interface')

        if not vima_interface:
            # Fallback: create a new interface if none provided (shouldn't happen in normal flow)
            print("⚠️ Warning: No shared VIMA interface found in state, creating new one...")
            vima_interface = VIMAInterface(
                modalities=["rgb"],
                debug=False,
                display_debug_window=False,
                hide_arm_rgb=False
            )

        actions: List[Action] = []
        vima_actions: List[VIMAAction] = []

        # Convert thinker tuples to VIMA actions
        if decision == 'SUCCESS' and pick_place_tuples:
            # Orientation defaults
            default_pick_orientation = (0.0, 0.0, 0.0)

            for i, (pick_pos, place_pos) in enumerate(pick_place_tuples):
                # Get rotation for this specific action
                rotation_deg = rotation_degrees_list[i] if i < len(rotation_degrees_list) else 0.0

                # Calculate place orientation based on rotation
                place_orientation = (0.0, 0.0, 0.0)  # Default: no rotation
                if rotation_deg != 0.0:
                    # Apply rotation about Z (yaw) in radians at placement
                    yaw = math.radians(rotation_deg)
                    place_orientation = (0.0, 0.0, yaw)

                # Create Action for state management
                action: Action = {
                    'pick_position': pick_pos,
                    'place_position': place_pos,
                    'pick_orientation': default_pick_orientation,
                    'place_orientation': place_orientation,
                }
                actions.append(action)

                # Create VIMAAction for execution
                vima_action = VIMAAction(
                    pose0_position=np.array(pick_pos),
                    pose0_rotation=np.array(default_pick_orientation),
                    pose1_position=np.array(place_pos),
                    pose1_rotation=np.array(place_orientation),
                    action_metadata={
                        "action_index": i,
                        "rotation_degrees": rotation_deg,
                        "reasoning": reasoning
                    }
                )
                vima_actions.append(vima_action)

                print(50*"=",f"VIMA actions: {vima_actions}",50*"=")

                print(f"Action {i+1}: Pick at {pick_pos}, Place at {place_pos}")
                if rotation_deg != 0.0:
                    print(f"  Rotation: {rotation_deg} degrees")
                else:
                    print("  Rotation: 0 degrees (no rotation)")

        # Execute actions through VIMA interface
        task_done = False
        execution_log = ""
        execution_results = []

        # Get oracle action for comparison before executing actions
        oracle_action = None
        if vima_interface and vima_actions:
            print("🔍 Getting oracle action for comparison...")

            # Get current observation for oracle (without resetting environment)
            try:
                # Temporarily disable image saving for oracle observation
                original_save_setting = vima_interface.save_observation_images
                vima_interface.save_observation_images = False
                current_obs = vima_interface.get_observation()
                vima_interface.save_observation_images = original_save_setting
                if current_obs and hasattr(current_obs, 'rgb_image'):
                    # Convert PIL image to dict format expected by oracle
                    obs_dict = {
                        'rgb': {
                            'front': np.array(current_obs.rgb_image),
                            'top': np.array(current_obs.rgb_image)  # Use same image for both views
                        }
                    }
                    oracle_action = vima_interface.get_oracle_action(obs_dict)
                    if oracle_action is not None:
                        print("✅ Retrieved oracle action for comparison")
                    else:
                        print("⚠️  Could not retrieve oracle action")
            except Exception as e:
                print(f"⚠️  Failed to get oracle action: {e}")

        # Execute actions based on configuration mode
        if config.mode == ActorMode.ORACLE and oracle_action is not None:
            print("🎯 ORACLE MODE: Executing VIMA oracle action...")
            print(f"Oracle action: {oracle_action}")

            # Execute the oracle action
            success, results = vima_interface.execute_action_sequence([oracle_action])

            task_done = success
            execution_results = results

            # Get updated observation after action execution
            updated_image = None
            try:
                # Temporarily disable image saving for observation update
                original_save_setting = vima_interface.save_observation_images
                vima_interface.save_observation_images = False
                updated_obs = vima_interface.get_observation()
                vima_interface.save_observation_images = original_save_setting
                if updated_obs and hasattr(updated_obs, 'rgb_image'):
                    updated_image = updated_obs.rgb_image
                    print("✅ Retrieved updated observation after oracle action execution")
            except Exception as e:
                print(f"⚠️  Failed to get updated observation after oracle action: {e}")

            # Build detailed execution log
            log_parts = []
            for i, result in enumerate(results):
                status = "SUCCESS" if result.get('success', False) else "FAILED"
                reward = result.get('reward', 0)
                log_parts.append(f"Oracle Action {i+1}: {status} (Reward: {reward})")

            execution_log = ("DEBUG MODE - ORACLE EXECUTION:\n" +
                           "\n".join(log_parts) +
                           f"\n\nOriginal Reasoning: {reasoning}")

            # Compare oracle with devised actions for analysis
            if vima_actions and len(vima_actions) > 0:
                print("\n🔍 Action Comparison (Oracle vs Devised):")
                print(f"Devised actions: {len(vima_actions)} actions")

                if hasattr(oracle_action, 'pose0_position') and hasattr(oracle_action, 'pose1_position'):
                    oracle_pick = getattr(oracle_action, 'pose0_position', None)
                    oracle_place = getattr(oracle_action, 'pose1_position', None)

                    devised_action = vima_actions[0]  # Compare with first devised action
                    devised_pick = getattr(devised_action, 'pose0_position', None)
                    devised_place = getattr(devised_action, 'pose1_position', None)

                    if oracle_pick is not None and devised_pick is not None:
                        pick_diff = np.linalg.norm(np.array(oracle_pick) - np.array(devised_pick))
                        print(f"Pick position difference: {pick_diff:.4f}")

                    if oracle_place is not None and devised_place is not None:
                        place_diff = np.linalg.norm(np.array(oracle_place) - np.array(devised_place))
                        print(f"Place position difference: {place_diff:.4f}")

                execution_log += f"\n\nDevised Actions: {len(vima_actions)} actions"
                for i, action in enumerate(vima_actions):
                    pick_pos = getattr(action, 'pose0_position', 'N/A')
                    place_pos = getattr(action, 'pose1_position', 'N/A')
                    execution_log += f"\n  Action {i+1}: Pick {pick_pos}, Place {place_pos}"

        elif config.mode == ActorMode.PIPELINE and vima_actions:
            print(f"Executing {len(vima_actions)} devised actions in VIMA environment...")
            success, results = vima_interface.execute_action_sequence(vima_actions)

            task_done = success
            execution_results = results

            # Get updated observation after action execution
            updated_image = None
            try:
                # Temporarily disable image saving for observation update
                original_save_setting = vima_interface.save_observation_images
                vima_interface.save_observation_images = False
                updated_obs = vima_interface.get_observation()
                vima_interface.save_observation_images = original_save_setting
                if updated_obs and hasattr(updated_obs, 'rgb_image'):
                    updated_image = updated_obs.rgb_image
                    print("✅ Retrieved updated observation after devised action execution")
            except Exception as e:
                print(f"⚠️  Failed to get updated observation after devised action: {e}")

            # Build detailed execution log
            log_parts = []
            for i, result in enumerate(results):
                status = "SUCCESS" if result.get('success', False) else "FAILED"
                reward = result.get('reward', 0)
                log_parts.append(f"Action {i+1}/{len(vima_actions)}: {status} (Reward: {reward})")

            execution_log = "VIMA Execution Results:\n" + "\n".join(log_parts) + f"\n\nReasoning: {reasoning}"

            # Compare devised actions with oracle action
            if oracle_action is not None:
                print("\n🔍 Action Comparison:")
                print(f"Devised actions: {len(vima_actions)} actions")
                print(f"Oracle action: {oracle_action}")

                # Try to compare action structures if possible
                if hasattr(oracle_action, 'pose0_position') and hasattr(oracle_action, 'pose1_position'):
                    oracle_pick = getattr(oracle_action, 'pose0_position', None)
                    oracle_place = getattr(oracle_action, 'pose1_position', None)

                    if vima_actions and len(vima_actions) > 0:
                        devised_action = vima_actions[0]  # Compare with first devised action
                        devised_pick = getattr(devised_action, 'pose0_position', None)
                        devised_place = getattr(devised_action, 'pose1_position', None)

                        if oracle_pick is not None and devised_pick is not None:
                            pick_diff = np.linalg.norm(np.array(oracle_pick) - np.array(devised_pick))
                            print(f"Pick position difference: {pick_diff:.4f}")

                        if oracle_place is not None and devised_place is not None:
                            place_diff = np.linalg.norm(np.array(oracle_place) - np.array(devised_place))
                            print(f"Place position difference: {place_diff:.4f}")

                execution_log += f"\n\nOracle Action: {oracle_action}"

            if success:
                print("✅ All actions executed successfully in VIMA environment")
            else:
                print("⚠️ Some actions failed in VIMA environment")
        else:
            execution_log = f"FAILED: No actions to execute. Reasoning: {reasoning}"
            print(f"❌ No actions to execute for prompt: {current_prompt}")

            # Even if no actions were executed, get current observation for reflector
            updated_image = None
            try:
                # Temporarily disable image saving for observation update
                original_save_setting = vima_interface.save_observation_images
                vima_interface.save_observation_images = False
                updated_obs = vima_interface.get_observation()
                vima_interface.save_observation_images = original_save_setting
                if updated_obs and hasattr(updated_obs, 'rgb_image'):
                    updated_image = updated_obs.rgb_image
                    print("✅ Retrieved current observation (no actions executed)")
            except Exception as e:
                print(f"⚠️  Failed to get current observation: {e}")

            # Note: Don't close the VIMA interface here as it's shared with other nodes
        # The graph_setup will handle cleanup after the entire pipeline completes

        output: ActorOutput = {
            'task_done': task_done,
            'execution_log': execution_log,
            'actions': actions,
            'oracle_action': oracle_action,  # Add oracle action for debugging
        }

        new_outputs = state.get('actor_output', {}).copy()
        new_outputs[current_prompt] = output

        print(f"\nActor Execution for '{current_prompt}':")
        print(execution_log)

        # Prepare return state update
        return_state = {
            'actor_output': new_outputs
        }

        # Include updated image if available
        if 'updated_image' in locals() and updated_image is not None:
            return_state['image'] = updated_image
            print("📸 Updated graph state with new observation image")

        return return_state

    except Exception as e:
        print(f"❌ Actor node error: {e}")
        import traceback
        traceback.print_exc()

        # Return a minimal state update to prevent pipeline failure
        current_prompt = state.get('current_prompt', 'unknown') if state else 'unknown'
        new_outputs = (state.get('actor_output', {}) if state else {}).copy()

        error_output: ActorOutput = {
            'task_done': False,
            'execution_log': f"ACTOR ERROR: {str(e)}",
            'actions': [],
            'oracle_action': None,
        }
        new_outputs[current_prompt] = error_output

        print(f"🔧 Actor node returning error state update: {new_outputs}")
        return {
            'actor_output': new_outputs
        }

    # Catch-all in case of any other issues
    print("🔧 Actor node reached end without proper return - this should not happen!")
    return {'actor_output': {}}
