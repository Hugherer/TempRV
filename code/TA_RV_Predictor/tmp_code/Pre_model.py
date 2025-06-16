import torch
from torch import nn, optim
import numpy as np

class EventTimePredictor(nn.Module):
    def __init__(self, event_dim, hidden_dim, num_layers=2):
        super(EventTimePredictor, self).__init__()
        self.lstm = nn.LSTM(event_dim + 1, hidden_dim, num_layers, batch_first=True)  # Add one for time input
        self.fc_event = nn.Linear(hidden_dim, event_dim)
        self.fc_time = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        event_out = self.fc_event(out)
        time_out = self.fc_time(out)
        return event_out, time_out

def one_hot_encode(events, num_classes):
    return np.eye(num_classes)[events]

def load_data(file_path, theta, seq_length=10):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            events = eval(line)
            times = [item[0] for item in events]
            events = [item[1] for item in events]
            data.append((times, events))
    
    max_event = max(max(events) for _, events in data) + 1
    normalized_data = []
    for times, events in data:
        #times_normalized = [(t - times[0]) / (times[-1] - times[0]) if len(times) > 1 else 0.0 for t in times]
        times_normalized = [(t - (min(times)-theta)) / ((max(times)+theta) - (min(times)-theta)) if len(times) > 1 else 0.0 for t in times]
        events_encoded = one_hot_encode(events, max_event)
        combined_input = np.hstack([events_encoded, np.array(times_normalized).reshape(-1, 1)])
        normalized_data.append(combined_input)
    
    return normalized_data, max_event

def train(model, data, save_path, epochs=200, learning_rate=0.01):
    criterion_event = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for sequence in data:
            inputs = torch.tensor(sequence[:-1], dtype=torch.float).unsqueeze(0)
            event_targets = torch.tensor(np.argmax(sequence[1:, :-1], axis=1), dtype=torch.long).unsqueeze(0)
            time_targets = torch.tensor(sequence[1:, -1:], dtype=torch.float).unsqueeze(0)
            
            optimizer.zero_grad()
            event_outputs, time_outputs = model(inputs)
            loss_event = criterion_event(event_outputs.view(-1, event_outputs.size(-1)), event_targets.view(-1))
            loss_time = criterion_time(time_outputs.view(-1, 1), time_targets.view(-1, 1))
            loss = loss_event + loss_time  # You may want to weight these losses differently
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path} successfully ")

    
def predict_sequence(model, totol_time, input_seq, output_len, num_classes):
    model.eval()
    predicted_events = []
    predicted_times = []
    with torch.no_grad():
        current_input = input_seq.copy()
        last_time = current_input[-1][-1]  # Last known time
        for i in range(output_len):
            #inputs = torch.tensor(current_input[-6:], dtype=torch.float).unsqueeze(0)
            inputs = torch.tensor(current_input[-6:], dtype=torch.float).unsqueeze(0)


            event_outputs, time_outputs = model(inputs)
            next_event_idx = torch.argmax(event_outputs[:, -1, :], dim=-1).item()
            next_time = time_outputs[:, -1, :].item() * (current_input[-1][-1] - current_input[0][-1]) + current_input[0][-1]  # Denormalize time
            next_event_one_hot = one_hot_encode([next_event_idx], num_classes)
            next_input = np.hstack([next_event_one_hot, [[next_time]]])
            predicted_events.append(next_event_idx)
            predicted_times.append(next_time*totol_time)
            #predicted_times.append(next_time)
            current_input = np.vstack([current_input, next_input])
    return list(zip(predicted_times, predicted_events))

def main():
    # Load your data; 一个是包含所有时间序列的列表 data，另一个是事件类别总数 num_classes（用于one-hot编码）
    theta = 10
    data, num_classes = load_data('input_data.txt',theta)

    # Initialize model  
    hidden_dim = 50
    model = EventTimePredictor(event_dim=num_classes, hidden_dim=hidden_dim)

    # Train model
    save_path = 'model_path'
    train(model, data, save_path)
    model.load_state_dict(torch.load(save_path))

    # Test input sequence
    test_input_sequence_list = [[[3, 1], [14, 2], [28, 5], [35, 6], [40, 6], [56, 7], [69, 7], [71, 8]],
                                [[91, 7], [86, 6], [70, 6], [68, 3], [53, 3], [45, 2], [35, 1], [20, 0]],
                                [[97, 9], [80, 9], [72, 8], [63, 7], [54, 7], [42, 3], [30, 2]],
                                [[0, 0], [15, 0], [24, 1], [39, 1], [48, 2], [51, 2], [65, 6], [76, 7], [87, 8]]]
    for i in range(4):
        test_input_sequence = test_input_sequence_list[i]
        # Extract only the event indices and times
        times, test_events = zip(*test_input_sequence)
        totol_time = max(times) - min(times) + 2*theta

        test_events = np.array(test_events)
        test_events_encoded = one_hot_encode(test_events, num_classes)
        times_normalized = [(t - (min(times)-theta)) / ((max(times)+theta) - (min(times)-theta)) if len(times) > 1 else 0.0 for t in times]
        
        test_input_combined = np.hstack([test_events_encoded, np.array(times_normalized).reshape(-1, 1)])

        # Predict future events based on the test input sequence
        predicted_events_and_times = predict_sequence(model, totol_time, test_input_combined, output_len=(10 - len(test_input_sequence)), num_classes=num_classes)

        # Combine the time and predicted events for output
        print(predicted_events_and_times)

if __name__ == '__main__':
    main()