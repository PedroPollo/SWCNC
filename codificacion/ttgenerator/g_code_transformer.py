import torch
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class GCodeDataset(Dataset):
    def __init__(self, coords_data, target_data, max_length=300):
        """
        Args:
            coords_data (list): Lista de coordenadas procesadas
            target_data (list): Lista de tokens procesados
            max_length (int): Longitud máxima para padding
        """
        self.coords = coords_data
        self.targets = target_data
        self.max_length = max_length

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # Convertir a tensores
        coord = torch.tensor(self.coords[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Aplicar padding
        coord_padded = self._pad_tensor(coord)
        target_padded = self._pad_tensor(target)

        return coord_padded, target_padded

    def _pad_tensor(self, tensor):
        """
        Hace padding del tensor hasta max_length
        """
        if len(tensor) > self.max_length:
            return tensor[:self.max_length]
        elif len(tensor) < self.max_length:
            padding_size = self.max_length - len(tensor)
            if tensor.dim() == 1:
                return torch.cat([tensor, torch.zeros(padding_size, dtype=tensor.dtype)])
            else:
                return torch.cat([tensor, torch.zeros(padding_size, tensor.size(-1), dtype=tensor.dtype)])
        return tensor
    

def create_dataloader(coords_data, tokens_data, batch_size=32, max_length=300, shuffle=True):
    dataset = GCodeDataset(coords_data, tokens_data, max_length=max_length)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

def tokenize_gcode(gcode, command_vocab, param_vocab):
    """
    Tokeniza un string de G-code en una lista de tokens.

    Args:
        gcode (str): String de G-code a tokenizar
        command_vocab (dict): Diccionario de comandos G y M
        param_vocab (dict): Diccionario de parámetros

    Returns:
        list: Lista de tuplas (tipo_token, valor)
    """
    lines = gcode.split('\n')
    tokens = []
    current_motion_command = None  # Para G00, G01, G02, G03

    for line in lines:
        line = line.strip()
        if not line or '(' in line:  # Ignorar comentarios y líneas vacías
            continue

        parts = line.split()
        command_in_line = False
        params_in_line = False

        # Procesar cada parte de la línea
        for part in parts:
            # Detectar comandos G o M
            if part.startswith(('G', 'M')) and part in command_vocab:
                command_token = ('CMD', command_vocab[part])
                tokens.append(command_token)
                command_in_line = True

                # Actualizar comando de movimiento actual si es G00, G01, G02 o G03
                if part in ['G00', 'G01', 'G02', 'G03']:
                    current_motion_command = command_token

            # Procesar parámetros
            elif any(part.startswith(param) for param in param_vocab):
                param_letter = part[0]
                try:
                    param_value = float(part[1:])

                    # Si es un parámetro de movimiento (X, Y) y no hay comando en la línea
                    if param_letter in ['X'] and not command_in_line:
                        # Añadir el último comando de movimiento si existe
                        if current_motion_command is not None:
                            tokens.append(current_motion_command)
                    elif param_letter in ['Y'] and not params_in_line and not command_in_line:
                        if current_motion_command is not None:
                            tokens.append(current_motion_command)

                    params_in_line = True

                    tokens.append(('PARAM', param_vocab[param_letter]))
                    tokens.append(('VALUE', param_value))
                except ValueError:
                    print(f"Advertencia: valor no válido para parámetro {part}")
                    continue

    return tokens

def transform_tokens(tokens):
    """
    Transforma los tokens a un formato numérico.

    Args:
        tokens (list): Lista de tuplas (tipo_token, valor)

    Returns:
        list: Lista de tuplas (tipo_numérico, índice, valor)
    """
    transformed_tokens = []

    for token_type, value in tokens:
        if token_type == 'CMD':
            transformed_tokens.append((1, value, 0))
        elif token_type == 'PARAM':
            transformed_tokens.append((2, value, 0))
        elif token_type == 'VALUE':
            transformed_tokens.append((3, 0, value))
    return transformed_tokens

def process_coordinates(figures, lone_lines, circles, arcs):
    """
    Procesa las coordenadas geométricas y las convierte en una lista plana.

    Args:
        figures (list): Lista de figuras geométricas
        lone_lines (list): Lista de líneas sueltas
        circles (list): Lista de círculos
        arcs (list): Lista de arcos

    Returns:
        list: Lista plana de coordenadas
    """
    coord_data = []

    # Procesar figuras geométricas
    if figures:
        flat_figures = np.array([
            point for figure in figures
            for line in figure for point in line
        ]).flatten()
        coord_data.extend(flat_figures)

    # Procesar líneas sueltas
    if lone_lines:
        line_data = np.array([
            point for line in lone_lines
            for point in line
        ]).flatten()
        coord_data.extend(line_data)

    # Procesar círculos
    if circles:
        circle_data = np.array([
            [c['center'][0], c['center'][1], c['radius']]
            for c in circles
        ]).flatten()
        coord_data.extend(circle_data)

    # Procesar arcos
    if arcs:
        arc_data = np.array([
            [a['center'][0], a['center'][1], a['radius'], a['start_angle'], a['end_angle']]
            for a in arcs
        ]).flatten()
        coord_data.extend(arc_data)

    return coord_data

def load_from_csv(csv_path, command_vocab, param_vocab):
    """
    Carga y procesa datos desde un archivo CSV.

    Args:
        csv_path (str): Ruta al archivo CSV
        command_vocab (dict): Diccionario de comandos G y M
        param_vocab (dict): Diccionario de parámetros

    Returns:
        tuple: (coordinates_data, tokens_data)
    """
    # Leer CSV
    df = pd.read_csv(csv_path)

    # Procesar coordenadas
    coordinates = []
    for _, row in df.iterrows():
        coord_dict = {
            'figures': eval(row['figures']) if not pd.isna(row['figures']) else [],
            'lone_lines': eval(row['lone_lines']) if not pd.isna(row['lone_lines']) else [],
            'circles': eval(row['circles']) if not pd.isna(row['circles']) else [],
            'arcs': eval(row['arcs']) if not pd.isna(row['arcs']) else []
        }
        coordinates.append(coord_dict)

    # Obtener G-codes
    gcodes = df['g_code'].tolist()

    # Procesar tokens y coordenadas
    tokens = []
    processed_coordinates = []

    for gcode, coord_data in zip(gcodes, coordinates):
        # Procesar G-code
        gcode_tokens = tokenize_gcode(gcode, command_vocab, param_vocab)
        transformed_tokens = transform_tokens(gcode_tokens)
        tokens.append(transformed_tokens)

        # Procesar coordenadas
        processed_coords = process_coordinates(
            coord_data['figures'],
            coord_data['lone_lines'],
            coord_data['circles'],
            coord_data['arcs']
        )
        processed_coordinates.append(processed_coords)

    return processed_coordinates, tokens

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GCodeTransformer(nn.Module):
    def __init__(self, seq_length=508, d_model=512, nhead=8, num_layers=6,dim_feedforward=2048, dropout=0.1, num_types=4, num_indices=20):
        super().__init__()

        # Ajustar tamaños de embeddings según los datos reales
        self.coord_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # Usar los tamaños correctos para los embeddings
        self.token_type_embedding = nn.Embedding(num_types, d_model)
        self.token_index_embedding = nn.Embedding(num_indices, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Decodificadores ajustados a los tamaños correctos
        self.type_decoder = nn.Linear(d_model, num_types)
        self.index_decoder = nn.Linear(d_model, num_indices)
        self.value_decoder = nn.Linear(d_model, 1)

    def forward(self, coords, target_tokens=None, mask=None):
        batch_size, seq_length = coords.size()
        coords = coords.unsqueeze(-1)
        x = self.coord_encoder(coords)

        if self.training and target_tokens is not None:
            types = target_tokens[:, :, 0].long()
            indices = target_tokens[:, :, 1].long()
            values = target_tokens[:, :, 2].unsqueeze(-1)

            # Asegurarse de que los índices estén dentro del rango
            types = torch.clamp(types, 0, self.token_type_embedding.num_embeddings - 1)
            indices = torch.clamp(indices, 0, self.token_index_embedding.num_embeddings - 1)

            type_emb = self.token_type_embedding(types)
            index_emb = self.token_index_embedding(indices)
            value_emb = self.coord_encoder(values)

            x = x + type_emb + index_emb + value_emb

        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x, mask)

        type_logits = self.type_decoder(transformer_output)
        index_logits = self.index_decoder(transformer_output)
        values = self.value_decoder(transformer_output)

        return type_logits, index_logits, values

    def get_token_embedding(self, token):
        type_idx = token[:, 0].long()
        index_idx = token[:, 1].long()
        value = token[:, 2]

        type_emb = self.token_type_embedding(type_idx)
        index_emb = self.token_index_embedding(index_idx)
        value_emb = self.coord_encoder(value.unsqueeze(-1))

        return type_emb + index_emb + value_emb

    def _is_end_token(self, token):
        # Implementar lógica para detectar token de fin de secuencia
        # Por ejemplo, si el tipo es 0 o si es un comando específico de finalización
        return (token[:, 0] == 0).any()

class GCodeTransformerTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

        # Modificar el learning rate y añadir scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Historiales
        self.loss_history = []
        self.best_loss = float('inf')

    def train_step(self, batch):
        self.model.train()
        coords, target_tokens = batch

        coords = coords.to(self.device)
        target_tokens = target_tokens.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass con mejor manejo de tipos
        type_logits, index_logits, values = self.model(coords, target_tokens)

        # Extraer objetivos y asegurar tipos correctos
        types = target_tokens[:, :, 0].long()
        indices = target_tokens[:, :, 1].long()
        target_values = target_tokens[:, :, 2].float()

        # Aplicar máscara para ignorar el padding
        mask = (types != 0)  # asumiendo que 0 es el valor de padding

        # Calcular pérdidas solo en elementos no-padding
        type_loss = F.cross_entropy(
            type_logits[mask].contiguous().view(-1, type_logits.size(-1)),
            types[mask].contiguous().view(-1)
        )

        index_loss = F.cross_entropy(
            index_logits[mask].contiguous().view(-1, index_logits.size(-1)),
            indices[mask].contiguous().view(-1)
        )

        value_loss = F.mse_loss(
            values[mask].contiguous().view(-1),
            target_values[mask].contiguous().view(-1)
        )

        # Balancear las pérdidas
        total_loss = type_loss + index_loss + 0.1 * value_loss

        total_loss.backward()

        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return total_loss.item()

    def predict(self, coords):
        self.model.eval()

        if len(coords.shape) == 1:
            coords = coords.unsqueeze(0)

        coords = coords.to(self.device)

        with torch.no_grad():
            type_logits, index_logits, values = self.model(coords)

            # Aplicar softmax para obtener probabilidades
            type_probs = F.softmax(type_logits, dim=-1)
            index_probs = F.softmax(index_logits, dim=-1)

            # Tomar los índices más probables
            types = torch.argmax(type_probs, dim=-1)
            indices = torch.argmax(index_probs, dim=-1)

            # Combinar las predicciones
            predictions = []
            for i in range(types.size(1)):
                type_val = types[0, i].item()
                index_val = indices[0, i].item()
                value_val = values[0, i, 0].item()

                # Ignorar predicciones de padding
                if type_val == 0:
                    continue

                predictions.append([type_val, index_val, value_val])

            return self.tokens_to_gcode(predictions)

    def tokens_to_gcode(self, tokens):
        gcode_lines = []
        current_line = []

        # Mapeos actualizados
        g_commands = {
            1: "G00", 2: "G01", 3: "G02", 4: "G03",
            5: "G04", 6: "G21", 7: "G40", 8: "G53",
            9: "G90", 10: "G91.1"
        }

        params = {
            1: "X", 2: "Y", 3: "I", 4: "J",
            5: "F", 6: "S", 7: "P"
        }

        for token in tokens:
            token_type = int(token[0])
            token_index = int(token[1])
            token_value = float(token[2])

            if token_type == 1:  # Comando
                if current_line:
                    gcode_lines.append(" ".join(current_line))
                    current_line = []

                if token_index in g_commands:
                    current_line.append(g_commands[token_index])

            elif token_type == 2:  # Parámetro
                if token_index in params:
                    param = params[token_index]
                    value = round(token_value, 3)
                    current_line.append(f"{param}{value}")

        if current_line:
            gcode_lines.append(" ".join(current_line))

        return "\n".join(gcode_lines)

def train_model(trainer, dataloader, num_epochs=100):
    print("Iniciando entrenamiento...")

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            loss = trainer.train_step(batch)
            epoch_losses.append(loss)

            if batch_idx % 10 == 0:
                print(f"Época {epoch+1}/{num_epochs} - Batch {batch_idx} - Loss: {loss:.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        trainer.loss_history.append(avg_loss)

        # Actualizar learning rate basado en la pérdida
        trainer.scheduler.step(avg_loss)

        # Guardar el mejor modelo
        if avg_loss < trainer.best_loss:
            trainer.best_loss = avg_loss
            torch.save(trainer.model.state_dict(), 'best_model.pt')

        print(f"Época {epoch+1} completada - Loss promedio: {avg_loss:.4f}")
        
# Primero, analicemos los rangos de valores en los datos
def analyze_data(dataloader):
    max_type = float('-inf')
    max_index = float('-inf')
    for coords, tokens in dataloader:
        types = tokens[:, :, 0]
        indices = tokens[:, :, 1]
        print(f"Types range: {types.min().item()} to {types.max().item()}")
        print(f"Indices range: {indices.min().item()} to {indices.max().item()}")
        max_type = max(max_type, types.max().item())
        max_index = max(max_index, indices.max().item())
        break  # Solo analizamos el primer batch
    return int(max_type) + 1, int(max_index) + 1

def process_coordinates2(test_input):
    """
    Procesa las coordenadas del formato `test_input` y las convierte en un tensor.
    """
    coord_data = []

    # Procesar figuras geométricas
    if test_input['figures']:
        flat_figures = np.array([
            point for figure in test_input['figures']
            for line in figure for point in line
        ]).flatten()
        coord_data.extend(flat_figures)

    # Procesar líneas sueltas
    if test_input['lone_lines']:
        line_data = np.array([
            point for line in test_input['lone_lines']
            for point in line
        ]).flatten()
        coord_data.extend(line_data)

    # Procesar círculos
    if test_input['circles']:
        circle_data = np.array([
            [c['center'][0], c['center'][1], c['radius']]
            for c in test_input['circles']
        ]).flatten()
        coord_data.extend(circle_data)

    # Procesar arcos
    if test_input['arcs']:
        arc_data = np.array([
            [a['center'][0], a['center'][1], a['radius'], a['start_angle'], a['end_angle']]
            for a in test_input['arcs']
        ]).flatten()
        coord_data.extend(arc_data)

    # Convertir la lista de coordenadas a un tensor
    coord_tensor = torch.tensor(coord_data, dtype=torch.float32)

    # Ajustar a longitud 508
    if coord_tensor.size(0) < 508:
        # Si el tensor tiene menos de 508 dimensiones, lo rellenamos con ceros
        padding = torch.zeros(508 - coord_tensor.size(0), dtype=torch.float32)
        coord_tensor = torch.cat((coord_tensor, padding), dim=0)
    elif coord_tensor.size(0) > 508:
        # Si el tensor tiene más de 508 dimensiones, lo truncamos
        coord_tensor = coord_tensor[:508]

    return coord_tensor