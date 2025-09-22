import jax
import jax.numpy as jnp


def get_device() -> jax.lib.xla_client.Device:
    """
    Returns the default JAX device (CPU or GPU) available on the system.
    """
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices found.")
    return devices[0]


def to_device(array: jnp.ndarray, device: jax.lib.xla_client.Device) -> jnp.ndarray:
    """
    Transfers a JAX array to the specified device.

    Args:
        array (jnp.ndarray): The JAX array to transfer.
        device (jax.lib.xla_client.Device): The target device.

    Returns:
        jnp.ndarray: The array on the specified device.
    """
    return jax.device_put(array, device)
