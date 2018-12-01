import psycopg2


class DataLoader():
    def __init__(self, config):
        self.connection = psycopg2.connect(user=config.get('user'),
                                           host=config.get('host'),
                                           port=config.get('port'))
        self.cursor = self.connection.cursor()

    def _get_values(self, primary_key_values):
        sql_command = ''
        self.cursor.execute(sql_command)
        records = self.cursor.fetchall()
        return records

    def _save_values(self, primary_keys_values, save_keys_values):
        pass

    def load_attributes(self, house_id):
        return self._get_values({'house_id': house_id})

    def save_attributes(self, house_id, attribute_keys_values):
        self._save_values({'house_id': house_id}, attribute_keys_values)

    def update_claim(self, house_id, claim):
        self._save_values({'house_id': house_id}, {'claim_amount': claim})

    def disconnect(self):
        if self.connection:
            self.cursor.close()
            self.connection.close()
