swagger_json = {
    "swagger": "2.0",
    "info": {
        "title": "Flask Swagger Example",
        "description": "API documentation for a simple Flask app",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": ["http"],
    "paths": {
        "/item/{item_id}": {
            "get": {
                "summary": "Get an item by ID",
                "parameters": [
                    {
                        "name": "item_id",
                        "in": "path",
                        "required": True,
                        "type": "integer"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "item": {"type": "string"}
                            }
                        }
                    },
                    "404": {
                        "description": "Item not found"
                    }
                }
            },
            "put": {
                "summary": "Update an item by ID",
                "parameters": [
                    {
                        "name": "item_id",
                        "in": "path",
                        "required": True,
                        "type": "integer"
                    },
                    {
                        "name": "body",
                        "in": "body",
                        "required": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"}
                            }
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Item updated"
                    },
                    "404": {
                        "description": "Item not found"
                    }
                }
            }
        },
        "/item": {
            "post": {
                "summary": "Create a new item",
                "parameters": [
                    {
                        "name": "body",
                        "in": "body",
                        "required": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"}
                            }
                        }
                    }
                ],
                "responses": {
                    "201": {
                        "description": "Item created"
                    },
                    "400": {
                        "description": "Invalid data"
                    }
                }
            }
        },
        "/historical_vol": {
            "get": {
                "summary": "Merton-Jump Model simulation for spot volatility estimations",
                "parameters": [
                    {
                        "name": "ric",
                        "in": "query",
                        "required": True,
                        "type": "string",
                        "description": "Ric"
                    },
                    {
                        "name": "years",
                        "in": "query",
                        "required": True,
                        "type": "integer",
                        "description": "Num of years in historical data."
                    },
                    {
                        "name": "scanning",
                        "in": "query",
                        "required": False,
                        "type": "string",
                        "description": "scanning different years spans for comparison."
                    },
                ],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "meta": {
                                    "type": "object",
                                    "properties": {
                                        "n_paths": {"type": "integer"},
                                        "start" : {"type": "string"},
                                        "end"   : {"type": "string"},
                                    }
                                },
                                "data":{
                                    "type": "object",
                                    "properties": {
                                        "mle_sigma": {"type": "number"},
                                        "sim_mean": {"type": "number"},
                                        "sim_p68" : {"type": "number"},
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid or missing parameters"
                    }
                }
            }
        },
        "/price_ranges": {
            "get": {
                "summary": "Fetch price ranges indicated by Open Interests of options market",
                "parameters": [
                    {
                        "name": "underlying",
                        "in": "query",
                        "required": True,
                        "type": "string",
                        "description": "BTC,ETH, etc"
                    },
                    {
                        "name": "atm_contracts",
                        "in": "query",
                        "required": False,
                        "type": "string",
                        "description": "collect also the ATM contract names"
                    },
                    {
                        "name": "update",
                        "in": "query",
                        "required": False,
                        "type": "string",
                        "description": "true or [not provide], indicating using cached file, or fetch in realtime."
                    },
                    
                ],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid or missing parameters"
                    }
                }
            }
        },
        "/pricing_options_from_spot": {
            "get": {
                "summary": "Calculate options contract prices according to a sequence of spot prices",
                "parameters": [
                    {
                        "name": "contracts",
                        "in": "query",
                        "required": True,
                        "type": "string",
                        "description": "BTC-240919-58000-C,BTC-240918-58500-P,BTC-240919-59000-C"
                    },
                    {
                        "name": "prange",
                        "in": "query",
                        "required": False,
                        "type": "string",
                        "description": "50000,60000,1000"
                    },
                    
                ],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid or missing parameters"
                    }
                }
            }
        }
    }
}


