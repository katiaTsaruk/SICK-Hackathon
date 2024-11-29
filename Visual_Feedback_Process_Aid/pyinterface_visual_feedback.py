import serial
import time


class UserFeedback:
    def __init__(self, port, baud_rate=115200):
        """
        Initialize the serial connection.
        :param port: Serial port (e.g., 'COM3', '/dev/ttyUSB0')
        :param baud_rate: Baud rate for serial communication
        """
        try:
            self.serial_connection = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)  # Allow time for the connection to stabilize
            if self.serial_connection.is_open:
                print(f"Connected to {port} at {baud_rate} baud.")
            else:
                raise ConnectionError(f"Failed to open serial port: {port}")
        except serial.SerialException as e:
            raise ConnectionError(f"Serial error: {e}")

    def send_command(self, command):
        """
        Send a command to the ESP32.
        :param command: Command string to send
        """
        if self.serial_connection.is_open:
            try:
                self.serial_connection.write((command + "\n").encode())  # Add newline for proper parsing
                self.serial_connection.flush()
                print(f"Sent command: {command}")
                time.sleep(0.1)  # Delay to ensure the ESP32 processes the command
            except Exception as e:
                raise IOError(f"Failed to send command '{command}': {e}")
        else:
            raise ConnectionError("Serial connection is not open.")

    def idle(self):
        """Set the device to idle mode (white pulsing)."""
        self.send_command("idle")

    def start_solder(self):
        """Set the device to start soldering mode (yellow constant)."""
        self.send_command("startsolder")

    def end_solder(self):
        """Set the device to end soldering mode (green flash then constant)."""
        self.send_command("endsolder")

    def error(self):
        """Set the device to error mode (red flashing)."""
        self.send_command("error")

    def close(self):
        """Close the serial connection."""
        if self.serial_connection.is_open:
            self.serial_connection.close()
            print("Serial connection closed.")


# Example usage
if __name__ == "__main__":
    try:
        # Replace 'COM3' with the correct port for your ESP32
        feedback = UserFeedback(port="COM3")
        
        # Test commands
        feedback.idle()            # Set to idle mode (white pulsing)
        time.sleep(5)              # Wait for 5 seconds
        feedback.start_solder()    # Set to start soldering mode (yellow constant)
        time.sleep(10)              # Wait for 3 seconds
        feedback.end_solder()      # Set to end soldering mode (green flash then constant)
        time.sleep(10)              # Wait for 5 seconds
        feedback.error()           # Set to error mode (red flashing)
        time.sleep(10)   
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure the connection is closed properly
        feedback.close()
