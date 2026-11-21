#include <tensorc/rl.h>
#include <tensorc/ndtensor.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct tc_dqn_agent {
    tc_dqn_config config;
    tc_dense_layer *online_hidden;
    tc_dense_layer *online_output;
    tc_dense_layer *target_hidden;
    tc_dense_layer *target_output;
    tc_tensor_nd *online_params[4];
    tc_tensor_nd_optimizer *optimizer;
    float *states;
    float *next_states;
    float *rewards;
    size_t *actions;
    unsigned char *dones;
    size_t replay_size;
    size_t replay_next;
    size_t train_steps;
    uint32_t rng;
};

static uint32_t dqn_random(tc_dqn_agent *agent) {
    uint32_t x = agent->rng ? agent->rng : 1u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    agent->rng = x;
    return x;
}

static int finite_vector(const float *values, size_t count) {
    if (!values) return 0;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

static int valid_env(const tc_env *env) {
    return env && env->observation_size > 0 && env->action_count > 0 &&
           env->reset && env->step;
}

int tc_env_reset(tc_env *env, uint32_t seed, float *observation_out) {
    if (!valid_env(env) || !observation_out ||
        !env->reset(env->user_data, seed, observation_out))
        return 0;
    return finite_vector(observation_out, env->observation_size);
}

int tc_env_step(tc_env *env, size_t action, float *observation_out,
                float *reward_out, int *terminated_out, int *truncated_out) {
    if (!valid_env(env) || action >= env->action_count || !observation_out ||
        !reward_out || !terminated_out || !truncated_out)
        return 0;
    float reward = 0.0f;
    int terminated = 0, truncated = 0;
    if (!env->step(env->user_data, action, observation_out, &reward,
                   &terminated, &truncated) ||
        !finite_vector(observation_out, env->observation_size) ||
        !isfinite(reward) || (terminated != 0 && terminated != 1) ||
        (truncated != 0 && truncated != 1))
        return 0;
    *reward_out = reward;
    *terminated_out = terminated;
    *truncated_out = truncated;
    return 1;
}

void tc_env_close(tc_env *env) {
    if (!env) return;
    if (env->close) env->close(env->user_data);
    env->close = NULL;
    env->reset = NULL;
    env->step = NULL;
}

static tc_tensor_nd *network_forward(tc_dense_layer *hidden_layer,
                                     tc_dense_layer *output_layer,
                                     const tc_tensor_nd *input) {
    tc_tensor_nd *hidden = tc_dense_layer_forward(hidden_layer, input);
    tc_tensor_nd *activated = hidden ? tc_tensor_nd_relu(hidden) : NULL;
    tc_tensor_nd_free(hidden);
    tc_tensor_nd *output = activated
                               ? tc_dense_layer_forward(output_layer, activated)
                               : NULL;
    tc_tensor_nd_free(activated);
    return output;
}

static void collect_params(tc_dqn_agent *agent) {
    agent->online_params[0] = tc_dense_layer_weight(agent->online_hidden);
    agent->online_params[1] = tc_dense_layer_bias(agent->online_hidden);
    agent->online_params[2] = tc_dense_layer_weight(agent->online_output);
    agent->online_params[3] = tc_dense_layer_bias(agent->online_output);
}

static int valid_config(const tc_dqn_config *c) {
    if (!c || c->observation_size == 0 || c->action_count < 2 ||
        c->hidden_size == 0 || c->replay_capacity == 0 || c->batch_size == 0 ||
        c->batch_size > c->replay_capacity || c->target_update_interval == 0 ||
        !(c->learning_rate > 0.0f) || !isfinite(c->learning_rate) ||
        c->discount_factor < 0.0f || c->discount_factor > 1.0f ||
        !isfinite(c->discount_factor))
        return 0;
    if (c->replay_capacity > SIZE_MAX / c->observation_size / sizeof(float) ||
        c->replay_capacity > SIZE_MAX / sizeof(size_t) ||
        c->replay_capacity > SIZE_MAX / sizeof(float) ||
        c->batch_size > SIZE_MAX / c->observation_size / sizeof(float) ||
        c->batch_size > SIZE_MAX / c->action_count / sizeof(float))
        return 0;
    return 1;
}

tc_dqn_agent *tc_dqn_agent_create(const tc_dqn_config *config) {
    if (!valid_config(config)) return NULL;
    tc_dqn_agent *agent = (tc_dqn_agent *)calloc(1, sizeof(*agent));
    if (!agent) return NULL;
    agent->config = *config;
    agent->rng = config->seed ? config->seed : 1u;

    agent->online_hidden = tc_dense_layer_create(
        config->observation_size, config->hidden_size, config->seed + 1u);
    agent->online_output = tc_dense_layer_create(
        config->hidden_size, config->action_count, config->seed + 2u);
    agent->target_hidden = tc_dense_layer_create(
        config->observation_size, config->hidden_size, config->seed + 3u);
    agent->target_output = tc_dense_layer_create(
        config->hidden_size, config->action_count, config->seed + 4u);
    agent->states = (float *)malloc(config->replay_capacity *
                                    config->observation_size * sizeof(float));
    agent->next_states = (float *)malloc(config->replay_capacity *
                                         config->observation_size * sizeof(float));
    agent->rewards = (float *)malloc(config->replay_capacity * sizeof(float));
    agent->actions = (size_t *)malloc(config->replay_capacity * sizeof(size_t));
    agent->dones = (unsigned char *)malloc(config->replay_capacity);
    if (!agent->online_hidden || !agent->online_output || !agent->target_hidden ||
        !agent->target_output || !agent->states || !agent->next_states ||
        !agent->rewards || !agent->actions || !agent->dones) {
        tc_dqn_agent_free(agent);
        return NULL;
    }
    collect_params(agent);
    agent->optimizer = tc_tensor_nd_optimizer_adam(agent->online_params, 4,
                                                   config->learning_rate);
    if (!agent->optimizer || !tc_dqn_sync_target(agent)) {
        tc_dqn_agent_free(agent);
        return NULL;
    }
    return agent;
}

void tc_dqn_agent_free(tc_dqn_agent *agent) {
    if (!agent) return;
    tc_tensor_nd_optimizer_free(agent->optimizer);
    tc_dense_layer_free(agent->online_hidden);
    tc_dense_layer_free(agent->online_output);
    tc_dense_layer_free(agent->target_hidden);
    tc_dense_layer_free(agent->target_output);
    free(agent->states);
    free(agent->next_states);
    free(agent->rewards);
    free(agent->actions);
    free(agent->dones);
    free(agent);
}

int tc_dqn_select_action(tc_dqn_agent *agent, const float *observation,
                         float epsilon, size_t *action_out) {
    if (!agent || !action_out || !finite_vector(observation,
                                                agent->config.observation_size) ||
        !isfinite(epsilon) || epsilon < 0.0f || epsilon > 1.0f)
        return 0;
    if ((float)dqn_random(agent) / 4294967296.0f < epsilon) {
        *action_out = (size_t)(dqn_random(agent) % agent->config.action_count);
        return 1;
    }
    const size_t shape[] = {1, agent->config.observation_size};
    tc_tensor_nd *input = tc_tensor_nd_from_data(shape, 2, observation, 0);
    tc_tensor_nd *q_values = input
                                 ? network_forward(agent->online_hidden,
                                                   agent->online_output, input)
                                 : NULL;
    const float *values = tc_tensor_nd_const_data(q_values);
    int ok = finite_vector(values, agent->config.action_count);
    if (ok) {
        size_t best = 0;
        for (size_t i = 1; i < agent->config.action_count; ++i)
            if (values[i] > values[best]) best = i;
        *action_out = best;
    }
    tc_tensor_nd_free(q_values);
    tc_tensor_nd_free(input);
    return ok;
}

int tc_dqn_remember(tc_dqn_agent *agent, const float *observation,
                    size_t action, float reward,
                    const float *next_observation, int done) {
    if (!agent || action >= agent->config.action_count || (done != 0 && done != 1) ||
        !isfinite(reward) ||
        !finite_vector(observation, agent->config.observation_size) ||
        !finite_vector(next_observation, agent->config.observation_size))
        return 0;
    const size_t index = agent->replay_next;
    const size_t offset = index * agent->config.observation_size;
    memcpy(agent->states + offset, observation,
           agent->config.observation_size * sizeof(float));
    memcpy(agent->next_states + offset, next_observation,
           agent->config.observation_size * sizeof(float));
    agent->actions[index] = action;
    agent->rewards[index] = reward;
    agent->dones[index] = (unsigned char)done;
    agent->replay_next = (index + 1) % agent->config.replay_capacity;
    if (agent->replay_size < agent->config.replay_capacity)
        ++agent->replay_size;
    return 1;
}

int tc_dqn_sync_target(tc_dqn_agent *agent) {
    if (!agent || !agent->online_hidden || !agent->online_output ||
        !agent->target_hidden || !agent->target_output)
        return 0;
    return tc_tensor_nd_copy_(tc_dense_layer_weight(agent->target_hidden),
                              agent->online_params[0]) &&
           tc_tensor_nd_copy_(tc_dense_layer_bias(agent->target_hidden),
                              agent->online_params[1]) &&
           tc_tensor_nd_copy_(tc_dense_layer_weight(agent->target_output),
                              agent->online_params[2]) &&
           tc_tensor_nd_copy_(tc_dense_layer_bias(agent->target_output),
                              agent->online_params[3]);
}

static void free_batch(tc_tensor_nd *states, tc_tensor_nd *next_states,
                       tc_tensor_nd *q_values, tc_tensor_nd *next_q_values,
                       tc_tensor_nd *targets, tc_tensor_nd *loss,
                       float *states_data, float *next_data, float *target_data) {
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(targets);
    tc_tensor_nd_free(next_q_values);
    tc_tensor_nd_free(q_values);
    tc_tensor_nd_free(next_states);
    tc_tensor_nd_free(states);
    free(states_data);
    free(next_data);
    free(target_data);
}

tc_dqn_train_status tc_dqn_train_step(tc_dqn_agent *agent, float *loss_out) {
    if (loss_out) *loss_out = 0.0f;
    if (!agent || !loss_out) return TC_DQN_TRAIN_ERROR;
    const tc_dqn_config *c = &agent->config;
    const size_t batch = c->batch_size;
    if (agent->replay_size < batch) return TC_DQN_TRAIN_NOT_READY;

    float *states_data = (float *)malloc(batch * c->observation_size * sizeof(float));
    float *next_data = (float *)malloc(batch * c->observation_size * sizeof(float));
    float *target_data = (float *)malloc(batch * c->action_count * sizeof(float));
    size_t *sampled_indices = (size_t *)malloc(batch * sizeof(size_t));
    if (!states_data || !next_data || !target_data || !sampled_indices) {
        free(states_data); free(next_data); free(target_data); free(sampled_indices);
        return TC_DQN_TRAIN_ERROR;
    }
    for (size_t row = 0; row < batch; ++row) {
        const size_t index = (size_t)(dqn_random(agent) % agent->replay_size);
        sampled_indices[row] = index;
        memcpy(states_data + row * c->observation_size,
               agent->states + index * c->observation_size,
               c->observation_size * sizeof(float));
        memcpy(next_data + row * c->observation_size,
               agent->next_states + index * c->observation_size,
               c->observation_size * sizeof(float));
    }
    const size_t batch_shape[] = {batch, c->observation_size};
    const size_t output_shape[] = {batch, c->action_count};
    tc_tensor_nd *states = tc_tensor_nd_from_data(batch_shape, 2, states_data, 0);
    tc_tensor_nd *next_states = tc_tensor_nd_from_data(batch_shape, 2, next_data, 0);
    tc_tensor_nd *q_values = states
                                 ? network_forward(agent->online_hidden,
                                                   agent->online_output, states)
                                 : NULL;
    tc_tensor_nd *next_q_values = next_states
                                      ? network_forward(agent->target_hidden,
                                                        agent->target_output,
                                                        next_states)
                                      : NULL;
    const float *current = tc_tensor_nd_const_data(q_values);
    const float *next = tc_tensor_nd_const_data(next_q_values);
    int ok = current && next &&
             finite_vector(current, batch * c->action_count) &&
             finite_vector(next, batch * c->action_count);
    if (ok) {
        memcpy(target_data, current, batch * c->action_count * sizeof(float));
        for (size_t row = 0; row < batch; ++row) {
            const size_t index = sampled_indices[row];
            size_t action = agent->actions[index];
            float max_next = next[row * c->action_count];
            for (size_t a = 1; a < c->action_count; ++a)
                if (next[row * c->action_count + a] > max_next)
                    max_next = next[row * c->action_count + a];
            const float td_target = agent->rewards[index] +
                (agent->dones[index] ? 0.0f : c->discount_factor * max_next);
            target_data[row * c->action_count + action] = td_target;
        }
    }
    tc_tensor_nd *targets = ok
                                ? tc_tensor_nd_from_data(output_shape, 2,
                                                         target_data, 0)
                                : NULL;
    tc_tensor_nd *loss = targets ? tc_tensor_nd_mse_loss(q_values, targets) : NULL;
    const float *value = tc_tensor_nd_const_data(loss);
    ok = value && isfinite(value[0]) && tc_tensor_nd_backward(loss) &&
         tc_tensor_nd_optimizer_step(agent->optimizer) == 4;
    if (ok) *loss_out = value[0];
    if (ok) {
        ++agent->train_steps;
        if (agent->train_steps % c->target_update_interval == 0)
            ok = tc_dqn_sync_target(agent);
    }
    free_batch(states, next_states, q_values, next_q_values, targets, loss,
               states_data, next_data, target_data);
    free(sampled_indices);
    return ok ? TC_DQN_TRAIN_UPDATED : TC_DQN_TRAIN_ERROR;
}

size_t tc_dqn_replay_size(const tc_dqn_agent *agent) {
    return agent ? agent->replay_size : 0;
}
