"""RF-Track configuration and imports."""
import RF_Track as rft

def show_versions():
    """Display RF-Track version and threading info."""
    print("RF-Track version:", rft.version)
    print("Max threads:", rft.max_number_of_threads)
