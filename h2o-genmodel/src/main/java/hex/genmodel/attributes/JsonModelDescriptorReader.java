package hex.genmodel.attributes;

import com.google.gson.*;
import hex.genmodel.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Utility class for extracting model details from JSON
 */
class JsonModelDescriptorReader {

    public static final String MODEL_DETAILS_FILE = "experimental/modelDetails.json";

    public static JsonObject parseModelJson(final MojoReaderBackend mojoReaderBackend) {

        try (BufferedReader fileReader = mojoReaderBackend.getTextFile(MODEL_DETAILS_FILE)) {
            final Gson gson = new GsonBuilder().create();

            return gson.fromJson(fileReader, JsonObject.class);
        } catch (IOException e) {
            throw new IllegalStateException("Could not read file inside MOJO " + MODEL_DETAILS_FILE, e);
        }
    }

    /**
     * Extracts a Table from H2O's model serialized into JSON.
     *
     * @param modelJson Full JSON representation of a model
     * @param tablePath Path in the given JSON to the desired table. Levels are dot-separated.
     * @return An instance of {@link Table}, if there was a table found by following the given path. Otherwise null.
     */
    public static Table extractTableFromJson(final JsonObject modelJson, final String tablePath) {
        Objects.requireNonNull(modelJson);
        JsonElement potentialTableJson = findInJson(modelJson, tablePath);
        if (potentialTableJson.isJsonNull()) {
            System.out.println(String.format("Failed to extract element '%s' MojoModel dump.",
                    tablePath));
            return null;
        }
        final JsonObject tableJson = potentialTableJson.getAsJsonObject();
        final int rowCount = tableJson.get("rowcount").getAsInt();

        final String[] columnHeaders;
        final Table.ColumnType[] columnTypes;
        final Object[][] data;


        // Extract column attributes
        final JsonArray columns = findInJson(tableJson, "columns").getAsJsonArray();
        final int columnCount = columns.size();
        columnHeaders = new String[columnCount];
        columnTypes = new Table.ColumnType[columnCount];

        for (int i = 0; i < columnCount; i++) {
            final JsonObject column = columns.get(i).getAsJsonObject();
            columnHeaders[i] = column.get("description").getAsString();
            columnTypes[i] = Table.ColumnType.extractType(column.get("type").getAsString());
        }


        // Extract data
        JsonArray dataColumns = findInJson(tableJson, "data").getAsJsonArray();
        data = new Object[columnCount][rowCount];
        for (int i = 0; i < columnCount; i++) {
            JsonArray column = dataColumns.get(i).getAsJsonArray();
            for (int j = 0; j < rowCount; j++) {
                final JsonPrimitive primitiveValue = column.get(j).getAsJsonPrimitive();

                switch (columnTypes[i]) {
                    case LONG:
                        data[i][j] = primitiveValue.getAsLong();
                        break;
                    case DOUBLE:
                        data[i][j] = primitiveValue.getAsDouble();
                        break;
                    case STRING:
                        data[i][j] = primitiveValue.getAsString();
                        break;
                }

            }
        }

        return new Table(tableJson.get("name").getAsString(), tableJson.get("description").getAsString(),
                new String[rowCount], columnHeaders, columnTypes, "", data);
    }


    private static final Pattern JSON_PATH_PATTERN = Pattern.compile("\\.|\\[|\\]");

    /**
     * Finds an element in GSON's JSON document representation
     *
     * @param jsonElement A (potentially complex) element to search in
     * @param jsonPath    Path in the given JSON to the desired table. Levels are dot-separated.
     *                    E.g. 'model._output.variable_importances'.
     * @return JsonElement, if found. Otherwise {@link JsonNull}.
     */
    private static JsonElement findInJson(JsonElement jsonElement, String jsonPath) {

        final String[] route = JSON_PATH_PATTERN.split(jsonPath);
        JsonElement result = jsonElement;

        for (String key : route) {
            key = key.trim();
            if (key.isEmpty())
                continue;

            if (result == null) {
                result = JsonNull.INSTANCE;
                break;
            }

            if (result.isJsonObject()) {
                result = ((JsonObject) result).get(key);
            } else if (result.isJsonArray()) {
                int value = Integer.valueOf(key) - 1;
                result = ((JsonArray) result).get(value);
            } else break;
        }

        return result;
    }
}
