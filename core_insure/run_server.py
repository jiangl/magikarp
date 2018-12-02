from server.magikarp import create_app

if __name__ == '__main__':
    config_file = open('./config.yaml', 'r')
    app = create_app(config_file)
    app.run()