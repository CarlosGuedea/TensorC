/* Entrena un agente DQN CartPole completamente en C11/TensorC. */
#include <tensorc/tensorc.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { OBSERVATIONS = 4, ACTIONS = 2, MAX_STEPS = 500, EPISODES = 250 };

typedef struct {
    double x, x_velocity, angle, angle_velocity;
    size_t steps;
    uint32_t rng;
} cartpole;

static double wall_time(void) {
    struct timespec now;
    timespec_get(&now, TIME_UTC);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static uint32_t random_u32(uint32_t *state) {
    uint32_t x = *state ? *state : 1u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double random_unit(uint32_t *state) {
    return (double)random_u32(state) / 4294967296.0;
}

static void observe(const cartpole *env, float observation[OBSERVATIONS]) {
    observation[0] = (float)(env->x / 2.4);
    observation[1] = (float)(env->x_velocity / 3.0);
    observation[2] = (float)(env->angle / 0.20943951);
    observation[3] = (float)(env->angle_velocity / 3.5);
}

static void reset(cartpole *env, uint32_t seed, float observation[OBSERVATIONS]) {
    env->rng = seed ? seed : 1u;
    env->x = (random_unit(&env->rng) - 0.5) * 0.08;
    env->x_velocity = (random_unit(&env->rng) - 0.5) * 0.08;
    env->angle = (random_unit(&env->rng) - 0.5) * 0.08;
    env->angle_velocity = (random_unit(&env->rng) - 0.5) * 0.08;
    env->steps = 0;
    observe(env, observation);
}

static void step(cartpole *env, size_t action,
                 float observation[OBSERVATIONS], float *reward,
                 int *terminated, int *truncated) {
    const double gravity = 9.8, mass_cart = 1.0, mass_pole = 0.1;
    const double total_mass = mass_cart + mass_pole, length = 0.5;
    const double pole_mass_length = mass_pole * length;
    const double force = action == 1 ? 10.0 : -10.0;
    const double tau = 0.02;
    const double costheta = cos(env->angle), sintheta = sin(env->angle);
    const double temp = (force + pole_mass_length * env->angle_velocity *
                         env->angle_velocity * sintheta) / total_mass;
    const double angle_acc = (gravity * sintheta - costheta * temp) /
        (length * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass));
    const double x_acc = temp - pole_mass_length * angle_acc * costheta / total_mass;

    env->x += tau * env->x_velocity;
    env->x_velocity += tau * x_acc;
    env->angle += tau * env->angle_velocity;
    env->angle_velocity += tau * angle_acc;
    ++env->steps;
    observe(env, observation);
    *reward = 1.0f;
    *terminated = fabs(env->x) > 2.4 || fabs(env->angle) > 0.20943951;
    *truncated = env->steps >= MAX_STEPS;
}

static int cartpole_reset(void *user_data, uint32_t seed,
                          float observation_out[OBSERVATIONS]) {
    if (!user_data || !observation_out) return 0;
    reset((cartpole *)user_data, seed, observation_out);
    return 1;
}

static int cartpole_step(void *user_data, size_t action,
                         float observation_out[OBSERVATIONS],
                         float *reward_out, int *terminated_out,
                         int *truncated_out) {
    if (!user_data || action >= ACTIONS || !observation_out || !reward_out ||
        !terminated_out || !truncated_out)
        return 0;
    step((cartpole *)user_data, action, observation_out, reward_out,
         terminated_out, truncated_out);
    return 1;
}

int main(void) {
    const tc_dqn_config config = {
        .observation_size = OBSERVATIONS,
        .action_count = ACTIONS,
        .hidden_size = 64,
        .replay_capacity = 10000,
        .batch_size = 64,
        .target_update_interval = 250,
        .learning_rate = 0.001f,
        .discount_factor = 0.99f,
        .seed = 42u,
    };
    tc_dqn_agent *agent = tc_dqn_agent_create(&config);
    if (!agent) {
        fprintf(stderr, "No se pudo crear el agente DQN.\n");
        return 1;
    }

    double first_window = 0.0, last_window = 0.0;
    size_t updates = 0, global_step = 0;
    const double started = wall_time();
    for (size_t episode = 0; episode < EPISODES; ++episode) {
        cartpole env = {0};
        float observation[OBSERVATIONS];
        tc_env environment = {
            .observation_size = OBSERVATIONS,
            .action_count = ACTIONS,
            .user_data = &env,
            .reset = cartpole_reset,
            .step = cartpole_step,
        };
        if (!tc_env_reset(&environment, (uint32_t)(100u + episode),
                          observation)) {
            fprintf(stderr, "No se pudo reiniciar CartPole.\n");
            tc_env_close(&environment);
            tc_dqn_agent_free(agent);
            return 1;
        }
        double episode_reward = 0.0;
        for (size_t t = 0; t < MAX_STEPS; ++t) {
            const float progress = fminf((float)global_step / 20000.0f, 1.0f);
            const float epsilon = 1.0f - 0.95f * progress;
            size_t action = 0;
            if (!tc_dqn_select_action(agent, observation, epsilon, &action)) {
                fprintf(stderr, "No se pudo seleccionar una acción.\n");
                tc_env_close(&environment);
                tc_dqn_agent_free(agent);
                return 1;
            }
            float next_observation[OBSERVATIONS], reward = 0.0f;
            int terminated = 0, truncated = 0;
            if (!tc_env_step(&environment, action, next_observation, &reward,
                             &terminated, &truncated)) {
                fprintf(stderr, "Falló el paso de CartPole.\n");
                tc_env_close(&environment);
                tc_dqn_agent_free(agent);
                return 1;
            }
            if (!tc_dqn_remember(agent, observation, action, reward,
                                 next_observation, terminated)) {
                fprintf(stderr, "No se pudo guardar la transición.\n");
                tc_env_close(&environment);
                tc_dqn_agent_free(agent);
                return 1;
            }
            float loss = 0.0f;
            const tc_dqn_train_status status = tc_dqn_train_step(agent, &loss);
            if (status == TC_DQN_TRAIN_ERROR) {
                fprintf(stderr, "Falló el paso de entrenamiento DQN.\n");
                tc_env_close(&environment);
                tc_dqn_agent_free(agent);
                return 1;
            }
            if (status == TC_DQN_TRAIN_UPDATED) ++updates;
            episode_reward += reward;
            ++global_step;
            for (size_t i = 0; i < OBSERVATIONS; ++i)
                observation[i] = next_observation[i];
            if (terminated || truncated) break;
        }
        tc_env_close(&environment);
        if (episode < 20) first_window += episode_reward;
        if (episode >= EPISODES - 20) last_window += episode_reward;
        if ((episode + 1) % 25 == 0 || episode + 1 == EPISODES)
            printf("episodio %3zu/%d | recompensa %.0f | epsilon %.3f\n",
                   episode + 1, EPISODES, episode_reward,
                   fmaxf(0.05f, 1.0f - 0.95f * fminf((float)global_step / 20000.0f, 1.0f)));
    }
    const double elapsed = wall_time() - started;
    printf("recompensa media primeros 20: %.1f | últimos 20: %.1f\n",
           first_window / 20.0, last_window / 20.0);
    printf("entrenamiento: %.3f s | pasos: %zu | actualizaciones: %zu\n",
           elapsed, global_step, updates);
    tc_dqn_agent_free(agent);
    return 0;
}
