#include <Adafruit_NeoPixel.h>

constexpr auto LED_PIN = 13;  // Pin connected to the NeoPixel LEDs
constexpr auto LED_NUM = 2;   // Number of LEDs

Adafruit_NeoPixel pixels(LED_NUM, LED_PIN, NEO_GRB + NEO_KHZ800);

enum DeviceState {
  IDLE,
  START_SOLDER,
  END_SOLDER,
  ERROR
};

DeviceState currentState = IDLE;

unsigned long stateStartTime = 0;
bool errorFlashingState = false; // Used to toggle red flashing

void setColor(uint8_t red, uint8_t green, uint8_t blue);
void handleState();

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  pixels.begin();
  stateStartTime = millis();

  Serial.println("ESP32 User Feedback Initialized");
}

void loop() {
  // Handle Serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove any extra spaces or newline characters

    if (command == "idle") {
      currentState = IDLE;
      stateStartTime = millis();
    } else if (command == "startsolder") {
      currentState = START_SOLDER;
      stateStartTime = millis();
    } else if (command == "endsolder") {
      currentState = END_SOLDER;
      stateStartTime = millis();
    } else if (command == "error") {
      currentState = ERROR;
      stateStartTime = millis();
    }
  }

  // Handle the current state
  handleState();
}

void handleState() {
  unsigned long currentTime = millis();

  // Declare brightness outside the switch statement
  float brightness = 0;

  switch (currentState) {
    case IDLE:
      // Pulsing white at 0.2 Hz
      brightness = 0.5 + 0.5 * sin(2 * 3.14159 * 0.2 * (currentTime - stateStartTime) / 1000.0);
      setColor(255 * brightness, 255 * brightness, 255 * brightness);
      break;

    case START_SOLDER:
      // Constant yellow
      setColor(255, 255, 0);
      break;

    case END_SOLDER:
      // Flash green briefly, then stay constant green
      if (currentTime - stateStartTime < 500) {
        setColor(0, 255, 0); // Flash green for 0.5 seconds
      } else {
        setColor(0, 255, 0); // Constant green
      }
      break;

    case ERROR:
      // Flash red at 2 Hz
      if ((currentTime / 250) % 2 == 0) { // Toggle every 250ms
        setColor(255, 0, 0); // Red on
      } else {
        setColor(0, 0, 0);   // Red off
      }
      break;
  }
}

void setColor(uint8_t red, uint8_t green, uint8_t blue) {
  for (int i = 0; i < LED_NUM; i++) {
    pixels.setPixelColor(i, pixels.Color(red, green, blue));
  }
  pixels.show();
}
