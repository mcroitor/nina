<?php

/**
 * Helper functions library for neural networks
 * 
 * Contains utilities for UUID generation, weight mutation, data conversion
 * between formats, and distance calculations between vectors.
 * 
 * @package mc
 * @author Mihail Croitor
 * @version 1.0
 */

namespace mc;

/**
 * Generates a random string in the specified format
 * 
 * Uses the given character set to create a string by pattern,
 * where 'x' is replaced with a random character from the set.
 * 
 * @param string $chars Character set for generation (default hex digits)
 * @param string $format String template, where 'x' - places for random characters
 * 
 * @return string Generated string by template
 * 
 * @example
 * ```php
 * $id = uuid(); // "a1b2-c3d4-e5f6-7890"
 * $custom = uuid("ABC", "x-x-x"); // "A-C-B"
 * ```
 */
function uuid(
    string $chars = "0123456789abcdef",
    string $format = "xxxx-xxxx-xxxx-xxxx"
): string {
    $result = "";

    for ($i = 0; $i < strlen($format); $i++) {
        if ($format[$i] == "x") {
            $result .= $chars[rand(0, strlen($chars) - 1)];
        } else {
            $result .= $format[$i];
        }
    }
    return $result;
}

/**
 * Mutates a value with given probability
 * 
 * Adds a random small change to the value with probability
 * determined by the precision parameter. Result is limited to range [0, 1].
 * 
 * @param float $value Original value for mutation
 * @param int $precision Mutation probability in percent (0-100)
 * 
 * @return float Mutated value in range [0, 1]
 * 
 * @example
 * ```php
 * $weight = 0.5;
 * $mutated = mutate($weight, 10); // 10% probability of change
 * ```
 */
function mutate(float $value, int $precision = 0): float {
    $rand = random_int(0, 100);
    if($rand < $precision) {
        $value += 0.001 * rand(-100, 100);
    }
    $value = limit($value);
    return $value;
}

/**
 * Limits a number to the range [0, 1]
 * 
 * If the number is less than 0, returns 0.
 * If the number is greater than 1, returns 1.
 * Otherwise returns the original number.
 * 
 * @param float $number Number to limit
 * 
 * @return float Limited number in range [0, 1]
 * 
 * @example
 * ```php
 * $limited = limit(-0.5); // 0.0
 * $limited = limit(1.5);  // 1.0
 * $limited = limit(0.7);  // 0.7
 * ```
 */
function limit(float $number): float
{
    if ($number < 0) {
        return 0.;
    }
    if ($number > 1) {
        return 1.;
    }
    return $number;
}

/**
 * Creates an array of specified size with default value
 * 
 * @param int $size Size of the array to create
 * @param float $value Value to fill the array with (default 0)
 * 
 * @return array Array of specified size filled with value
 * 
 * @example
 * ```php
 * $zeros = array_create(5);        // [0, 0, 0, 0, 0]
 * $ones = array_create(3, 1.0);    // [1.0, 1.0, 1.0]
 * ```
 */
function array_create(int $size, float $value = 0.): array {
    $result = [];
    $result = array_pad($result, $size, $value);
    return $result;
}

/**
 * Creates an array of specified size filled with random float values in range [0, 1]
 * 
 * @param int $size Size of the array to create
 * 
 * @return array Array of specified size filled with random values
 */
function array_random(int $size): array {
    $result = [];
    for ($i = 0; $i < $size; $i++) {
        $result[$i] = rand(0, 100) / 100.0;
    }
    return $result;
}

/**
 * Prints array contents to console
 * 
 * Prints each array element separated by space and adds
 * a newline at the end.
 * 
 * @param array $array Array to output
 * 
 * @return void
 * 
 * @example
 * ```php
 * print_array([1.5, 2.3, 0.8]); // Outputs: "1.5 2.3 0.8 \n"
 * ```
 */
function print_array(array $array): void {
    foreach ($array as $value) {
        echo "{$value} ";
    }
    echo PHP_EOL;
}

/**
 * Converts a character to a float from 0 to 1
 * 
 * Converts the ASCII code of a character to a normalized value
 * in range [0, 1] by dividing by 255.
 * 
 * @param string $char Character to convert
 * 
 * @return float Normalized ASCII code value (0-1)
 * 
 * @example
 * ```php
 * $normalized = char2float('A'); // 65/255 ≈ 0.255
 * $normalized = char2float(' '); // 32/255 ≈ 0.125
 * ```
 */
function char2float(string $char): float {
    return ord($char) / 255;
}

/**
 * Converts a float to a character
 * 
 * Converts a normalized value (0-1) to a character
 * by multiplying by 255 and converting to ASCII code.
 * 
 * @param float $value Normalized value (0-1)
 * 
 * @return string Character corresponding to the ASCII code
 * 
 * @example
 * ```php
 * $char = float2char(0.255); // Approximately 'A' (code 65)
 * $char = float2char(0.125); // Approximately ' ' (code 32)
 * ```
 */
function float2char(float $value): string {
    return chr(intval($value * 255));
}

/**
 * Converts an array of normalized values to a string
 * 
 * Converts each array element to a character and combines them into a string.
 * 
 * @param array $array Array of normalized values (0-1)
 * 
 * @return string String created from character array
 * 
 * @example
 * ```php
 * $string = array2string([0.255, 0.125, 0.3]); // Creates string from characters
 * ```
 */
function array2string(array $array): string {
    $result = "";
    foreach ($array as $value) {
        $result .= float2char($value);
    }
    return $result;
}

/**
 * Converts a string to an array of normalized values
 * 
 * Converts each character of the string to a normalized value (0-1).
 * If the specified array size is larger than string length, pads with zeros.
 * 
 * @param string $string String to convert
 * @param int $array_size Desired array size (0 = string length)
 * 
 * @return array Array of normalized character values
 * 
 * @example
 * ```php
 * $array = string2array("ABC");    // [0.255, 0.258, 0.262]
 * $array = string2array("AB", 5);  // [0.255, 0.258, 0, 0, 0]
 * ```
 */
function string2array(string $string, int $array_size = 0): array {
    $result = [];
    if($array_size == 0) {
        $array_size = strlen($string);
    }
    $len = strlen($string) < $array_size ? strlen($string) : $array_size;
    for ($i = 0; $i < $len; $i++) {
        $result[] = char2float($string[$i]);
    }
    for ($i = $len; $i < $array_size; $i++) {
        $result[] = 0;
    }
    return $result;
}

/**
 * Calculates Euclidean distance between two vectors or strings
 * 
 * If strings are passed, automatically converts them to arrays.
 * Calculates the square root of the sum of squared differences of elements.
 * 
 * @param array|string $a First vector or string
 * @param array|string $b Second vector or string
 * 
 * @return float Euclidean distance between vectors
 * 
 * @example
 * ```php
 * $distance = euclidean_distance([1, 2], [4, 6]); // sqrt((1-4)² + (2-6)²) = 5
 * $distance = euclidean_distance("AB", "CD");     // Distance between strings
 * ```
 */
function euclidean_distance(array|string $a, array|string $b): float {
    if(is_string($a)) {
        $a = string2array($a);
    }
    if(is_string($b)) {
        $b = string2array($b);
    }
    $result = 0;
    for ($i = 0; $i < count($a); $i++) {
        $result += ($a[$i] - $b[$i]) ** 2;
    }
    return sqrt($result);
}

/**
 * Calculates Manhattan distance between two vectors or strings
 * 
 * If strings are passed, automatically converts them to arrays.
 * Calculates the sum of absolute differences of corresponding elements.
 * Arrays must have the same size.
 * 
 * @param array|string $a First vector or string
 * @param array|string $b Second vector or string
 * 
 * @return float Manhattan distance between vectors, -1 on error
 * 
 * @example
 * ```php
 * $distance = manhattan_distance([1, 2], [4, 6]); // |1-4| + |2-6| = 7
 * $distance = manhattan_distance("AB", "CD");     // Distance between strings
 * ```
 */
function manhattan_distance(array|string $a, array|string $b): float {
    $result = 0;
    if(is_string($a)) {
        $a = string2array($a);
    }
    if(is_string($b)) {
        $b = string2array($b);
    }
    if(count($a) != count($b)) {
        echo "[error] arrays must have the same size\n";
        echo "[debug] a: " . json_encode($a) . ", size: " . count($a) . "\n";
        echo "[debug] b: " . json_encode($b) . ", size: " . count($b) . "\n";
        return -1;
    }
    for ($i = 0; $i < count($a); $i++) {
        $result += abs($a[$i] - $b[$i]);
    }
    return $result;
}