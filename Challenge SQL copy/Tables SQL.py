import sqlite3

# Crear conexión y cursor
conn = sqlite3.connect("base_de_Datos.db")
cursor = conn.cursor()

# Crear tablas con SQL corregido
cursor.execute("""
CREATE TABLE Proveedores (
    proveedor_id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    direccion TEXT,
    ciudad TEXT,
    provincia TEXT
);
""")

cursor.execute("""
CREATE TABLE Categorias (
    categoria_id INTEGER PRIMARY KEY,
    nombre TEXT
);
""")

cursor.execute("""
CREATE TABLE Piezas (
    pieza_id INTEGER PRIMARY KEY,
    nombre TEXT,
    color TEXT,
    precio REAL,
    categoria_id INTEGER,
    FOREIGN KEY (categoria_id) REFERENCES Categorias(categoria_id) ON DELETE SET NULL ON UPDATE SET NULL
);
""")

cursor.execute("""
CREATE TABLE Pedidos (
    suministro_id INTEGER PRIMARY KEY AUTOINCREMENT,
    proveedor_id INTEGER,
    pieza_id INTEGER,
    cantidad INTEGER,
    fecha DATE,
    FOREIGN KEY (proveedor_id) REFERENCES Proveedores(proveedor_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (pieza_id) REFERENCES Piezas(pieza_id) ON DELETE CASCADE ON UPDATE CASCADE,
    CHECK (cantidad > 0)
    UNIQUE (proveedor_id,pieza_id,fecha)
);
""")

# Insertar datos en Proveedores
cursor.execute("INSERT INTO Proveedores (proveedor_id,nombre,direccion,ciudad,provincia) VALUES (10054, 'Proveedor A', 'Calle Falsa 123', 'CDMX', 'CDMX')")

# Insertar datos en Categorias
cursor.execute("INSERT INTO Categorias (categoria_id,nombre) VALUES (1, 'Electrónica')")

# Insertar datos en Piezas
cursor.execute("INSERT INTO Piezas (pieza_id,nombre,color,precio,categoria_id) VALUES (10178, 'Resistor', 'Marrón', 0.15, 1)")

# Insertar datos en Pedidos
cursor.execute("INSERT INTO Pedidos (proveedor_id,pieza_id,cantidad,fecha) VALUES (10054,101, 500, '2025-04-10')")

conn.commit()