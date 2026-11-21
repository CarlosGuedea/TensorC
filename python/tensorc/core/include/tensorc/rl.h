#ifndef TENSORC_RL_H
#define TENSORC_RL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tc_dqn_agent tc_dqn_agent;

/* Caller-owned C environment with fixed-size vector observations/actions. */
typedef struct {
    size_t observation_size;
    size_t action_count;
    void *user_data;
    int (*reset)(void *user_data, uint32_t seed, float *observation_out);
    int (*step)(void *user_data, size_t action, float *observation_out,
                float *reward_out, int *terminated_out, int *truncated_out);
    void (*close)(void *user_data);
} tc_env;

typedef enum {
    TC_DQN_TRAIN_ERROR = -1,
    TC_DQN_TRAIN_NOT_READY = 0,
    TC_DQN_TRAIN_UPDATED = 1
} tc_dqn_train_status;

typedef struct {
    size_t observation_size;
    size_t action_count;
    size_t hidden_size;
    size_t replay_capacity;
    size_t batch_size;
    size_t target_update_interval;
    float learning_rate;
    float discount_factor;
    uint32_t seed;
} tc_dqn_config;

/* Callback returns 1 on success and 0 on failure; close is optional. */
int tc_env_reset(tc_env *env, uint32_t seed, float *observation_out);
int tc_env_step(tc_env *env, size_t action, float *observation_out,
                float *reward_out, int *terminated_out, int *truncated_out);
void tc_env_close(tc_env *env);

tc_dqn_agent *tc_dqn_agent_create(const tc_dqn_config *config);
void tc_dqn_agent_free(tc_dqn_agent *agent);

/* epsilon must be in [0,1]; returns 1 on success and 0 on invalid input. */
int tc_dqn_select_action(tc_dqn_agent *agent, const float *observation,
                         float epsilon, size_t *action_out);

/* Stores one transition; done must be 0 or 1. */
int tc_dqn_remember(tc_dqn_agent *agent, const float *observation,
                    size_t action, float reward,
                    const float *next_observation, int done);

/* Returns ERROR, NOT_READY until replay is full enough, or UPDATED. */
tc_dqn_train_status tc_dqn_train_step(tc_dqn_agent *agent, float *loss_out);
int tc_dqn_sync_target(tc_dqn_agent *agent);
size_t tc_dqn_replay_size(const tc_dqn_agent *agent);

#ifdef __cplusplus
}
#endif

#endif
