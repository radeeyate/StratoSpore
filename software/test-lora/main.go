package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode"

	"github.com/joho/godotenv"
	zmq "github.com/pebbe/zmq4"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type Log struct {
	Timestamp          time.Time `bson:"timestamp"`
	Temperature        float64   `bson:"temperature"`
	Humidity           float64   `bson:"humidity"`
	Pressure           float64   `bson:"pressure"`
	Latitude           float64   `bson:"latitude"`
	Longitude          float64   `bson:"longitude"`
	Altitude           float64   `bson:"altitude"`
	UVIndex            float64   `bson:"uvindex"`
	BatteryVoltage     float64   `bson:"batteryvoltage"`
	CPUUsage           float64   `bson:"cpuusage"`
	MemoryUsage        float64   `bson:"memoryusage"`
	DiskUsage          float64   `bson:"diskusage"`
	PicoCPUTemperature float64   `bson:"picocputemp"`
	PiCPUTemperature   float64   `bson:"picputemp"`
}

var (
	client           *mongo.Client
	database         *mongo.Database
	logs             *mongo.Collection
	ctx              context.Context
	connectionString string
)

const (
	zmqEndpoint          = "tcp://localhost:5555"
	zmqSubscriptionTopic = ""
)

func main() {
	err := godotenv.Load()
	if err != nil {
		panic("Error loading .env file")
	}

	connectionString = os.Getenv("CONNECTION_STRING")

	connect()
	ctx = context.Background()

	log.Println("Starting LoRa ZMQ listener...")

	context, err := zmq.NewContext()
	if err != nil {
		log.Fatalf("Error creating ZMQ context: %v", err)
	}
	defer context.Term()

	subscriber, err := context.NewSocket(zmq.SUB)
	if err != nil {
		log.Fatalf("Error creating ZMQ subscriber socket: %v", err)
	}
	defer subscriber.Close()

	log.Printf("Connecting to ZMQ publisher at %s", zmqEndpoint)
	err = subscriber.Connect(zmqEndpoint)
	if err != nil {
		log.Fatalf("Error connecting to ZMQ publisher: %v", err)
	}

	err = subscriber.SetSubscribe(zmqSubscriptionTopic)
	if err != nil {
		log.Fatalf("Error subscribing to topic '%s': %v", zmqSubscriptionTopic, err)
	}
	log.Printf("Subscribed to topic: '%s' (empty means all)", zmqSubscriptionTopic)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	done := make(chan bool, 1)

	go func() {
		sig := <-sigChan
		log.Printf("Received signal: %s. Shutting down...", sig)
		done <- true
	}()

	log.Println("Listening for messages...")

ReceiveLoop:
	for {
		select {
		case <-done:
			break ReceiveLoop
		default:
			msgBytes, err := subscriber.RecvBytes(0)
			if err != nil {
				if zmq.AsErrno(err) == zmq.ETERM {
					log.Println("ZMQ context terminated, exiting receive loop.")
				} else {
					log.Printf("Error receiving message: %v", err)
				}
				break ReceiveLoop
			}

			message := string(msgBytes)
			timestamp := time.Now().Format("2006-01-02 15:04:05")
			fmt.Printf("[%s] Received: %s\n", timestamp, message)

			decodedMessage, err := decodeMessage(message)
			if err != nil {
				log.Printf("Error decoding message: %v", err)
				continue
			}

			insertLog(decodedMessage)
		}
	}

	log.Println("LoRa ZMQ listener stopped.")
}

func connect() {
	clientOptions := options.Client().
		ApplyURI(connectionString)
	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		log.Fatal(err)
	}

	err = client.Ping(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}

	database = client.Database("stratospore-test")

	logs = database.Collection("logs")
}

func decodeMessage(message string) (Log, error) {
	var logData Log

	cleanedMessage := strings.TrimLeftFunc(message, func(r rune) bool {
		return !('0' <= r && r <= '9' || r == '.' || r == '-' || r == '|' || unicode.IsSpace(r))
	})

	parts := strings.Split(cleanedMessage, "|")
	if len(parts) != 13 {
		return logData, fmt.Errorf("invalid message format: expected 13 parts, got %d. Message: '%s'", len(parts), message)
	}

	var err error
	logData.Temperature, err = strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing temperature from '%s': %w", parts[0], err)
	}

	logData.Humidity, err = strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing humidity from '%s': %w", parts[1], err)
	}

	logData.Pressure, err = strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing pressure from '%s': %w", parts[2], err)
	}

	logData.Latitude, err = strconv.ParseFloat(strings.TrimSpace(parts[3]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing latitude from '%s': %w", parts[3], err)
	}

	logData.Longitude, err = strconv.ParseFloat(strings.TrimSpace(parts[4]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing longitude from '%s': %w", parts[4], err)
	}

	logData.Altitude, err = strconv.ParseFloat(strings.TrimSpace(parts[5]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing altitude from '%s': %w", parts[5], err)
	}

	logData.UVIndex, err = strconv.ParseFloat(strings.TrimSpace(parts[6]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing uv index from '%s': %w", parts[6], err)
	}

	logData.BatteryVoltage, err = strconv.ParseFloat(strings.TrimSpace(parts[7]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing battery voltage from '%s': %w", parts[7], err)
	}

	logData.CPUUsage, err = strconv.ParseFloat(strings.TrimSpace(parts[8]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing CPU usage from '%s': %w", parts[8], err)
	}
	
	logData.MemoryUsage, err = strconv.ParseFloat(strings.TrimSpace(parts[9]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing memory usage from '%s': %w", parts[9], err)
	}

	logData.DiskUsage, err = strconv.ParseFloat(strings.TrimSpace(parts[10]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing disk usage from '%s': %w", parts[10], err)
	}

	logData.PicoCPUTemperature, err = strconv.ParseFloat(strings.TrimSpace(parts[11]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing pico cpu temperature from '%s': %w", parts[11], err)
	}

	logData.PiCPUTemperature, err = strconv.ParseFloat(strings.TrimSpace(parts[12]), 64)
	if err != nil {
		return logData, fmt.Errorf("error parsing pi cpu temperature from '%s': %w", parts[12], err)
	}

	logData.Timestamp = time.Now()

	return logData, nil
}

func insertLog(logData Log) {
	_, err := logs.InsertOne(ctx, logData)
	if err != nil {
		log.Printf("Error inserting log: %v", err)
	} else {
		log.Println("Log inserted successfully.")
	}
}
