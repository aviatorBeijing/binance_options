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
        }
    }
}


