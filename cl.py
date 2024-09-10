import tensorflow as tf
import numpy as np

# Define cosine similarity function
def cosine_similarity(x, y):
    """
    Compute cosine similarity between two tensors.

    :param x: Tensor of shape [batch_size, hidden_dim]
    :param y: Tensor of shape [batch_size, hidden_dim]
    :return: Cosine similarity between x and y
    """
    x_norm = tf.nn.l2_normalize(x, axis=-1)
    y_norm = tf.nn.l2_normalize(y, axis=-1)
    return tf.reduce_sum(tf.multiply(x_norm, y_norm), axis=-1)  # Element-wise product along the hidden_dim axis

# Define contrastive loss function using TensorFlow operations
def contrastive_loss(outputs, y_src, tau):
    """
    Compute contrastive loss for the batch by taking the log of each fraction separately and summing over all logs.

    :param outputs: Feature representations (outputs_fin_source), shape [batch_size, hidden_dim]
    :param y_src: Labels (sentiment polarity), shape [batch_size, n_class]
    :param tau: Temperature parameter for contrastive loss scaling
    :return: Scalar contrastive loss value
    """
    batch_size = tf.shape(outputs)[0]  # Get the batch size

    def compute_loss_for_instance(i):
        vi = outputs[i]  # Anchor instance (output)
        yi = y_src[i]  # Anchor label (sentiment polarity)

        # Compute cosine similarities between the anchor (vi) and all other instances (outputs)
        sim_all = cosine_similarity(tf.expand_dims(vi, axis=0), outputs) / tau  # Shape: [batch_size]

        # Find all instances with the same sentiment polarity as the anchor
        mask_same_class = tf.equal(tf.argmax(y_src, 1), tf.argmax(yi))  # True if same class
        mask_same_class = tf.cast(mask_same_class, tf.float32)  # Convert boolean mask to float32

        # Create a mask to exclude the similarity of vi with itself (set it to 0)
        mask_not_self = 1.0 - tf.one_hot(i, depth=batch_size)  # Shape: [batch_size]

        # Exclude self from both the same class mask and similarities
        mask_pos = mask_same_class * mask_not_self  # Exclude the anchor itself from the positive mask

        # Apply the mask to exclude the self-similarity, compute sum
        sim_all_not_self = sim_all * mask_not_self  # Shape: [batch_size]
        sum_all = tf.reduce_sum(tf.exp(sim_all_not_self))  # Sum of exp(similarity) for all pairs (excluding self)

        # Extract positive similarities (those with the same sentiment polarity as vi, excluding self)
        sim_pos = tf.exp(sim_all) * mask_pos  # Apply mask to get only positive pairs

        # Now take the log of the fraction for each positive pair and accumulate the total loss
        log_fractions = mask_pos * tf.math.log(sim_pos / (sum_all + 1e-10))  # Log of each fraction

        # Sum all the logs for the current instance and add it to the total loss
        loss_for_instance = -tf.reduce_sum(log_fractions) / (tf.reduce_sum(mask_pos) + 1e-10)  # Avoid division by zero
        return loss_for_instance

    # Use tf.map_fn to apply the function over the batch
    loss_per_instance = tf.map_fn(compute_loss_for_instance, tf.range(batch_size), dtype=tf.float32)

    # Return the mean loss over the batch
    return tf.reduce_mean(loss_per_instance)

# Testing the functions
def test_contrastive_loss():
    # Create random data for testing
    batch_size = 20
    hidden_dim = 768
    num_classes = 3
    tau = 0.9

    # Randomly generate feature representations (outputs)
    outputs = tf.random.normal([batch_size, hidden_dim], mean=0, stddev=1)

    # Randomly generate one-hot encoded labels for the batch
    y_src = tf.one_hot(np.random.randint(0, num_classes, size=batch_size), depth=num_classes)

    # Compute the contrastive loss
    loss_value = contrastive_loss(outputs, y_src, tau)

    # Run the session to get the result
    with tf.compat.v1.Session() as sess:  # Use tf.compat.v1 for TensorFlow 1.x functions
        sess.run(tf.compat.v1.global_variables_initializer())
        loss_value_np = sess.run(loss_value)
        print(f"Contrastive Loss: {loss_value_np}")

# Run the test
if __name__ == "__main__":
    test_contrastive_loss()