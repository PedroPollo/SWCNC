from g_code_transformer import *

command_vocab = {
    'G00': 1, 'G01': 2, 'G02': 3, 'G03': 4, 'G04': 5,
    'G21': 6, 'G40': 7, 'G53': 8, 'G90': 9, 'G91.1': 10,
    'M03': 11, 'M05': 12, 'M30': 13, 'M801': 14, 'M802': 15
}
param_vocab = {
    'X': 1, 'Y': 2, 'I': 3, 'J': 4, 'F': 5,
    'S': 6, 'P': 7
}

def obtener_codigo(input):
    coords_data, tokens_data = load_from_csv('output.csv',command_vocab,param_vocab)
    max_coord_len = max(len(coord) for coord in coords_data)
    max_token_len = max(len(token) for token in tokens_data)
    max_length = max(max_coord_len, max_token_len)
    dataloader = create_dataloader(coords_data, tokens_data,batch_size=32,max_length=max_length)
    num_types, num_indices = analyze_data(dataloader)
    model = GCodeTransformer(seq_length= 508,num_types=num_types,num_indices=num_indices)
    device = torch.device('cpu')
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.to(device)
    
    trainer = GCodeTransformerTrainer(model)
    
    cords = process_coordinates2(input)
    cords = cords.unsqueeze(0)
    
    generated_gcode = trainer.predict(cords)
    return generated_gcode